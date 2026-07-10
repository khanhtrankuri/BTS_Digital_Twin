#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.geometry_losses import (
    edge_weighted_l1_loss,
    gaussian_scale_regularization,
    get_loss_weights,
    normal_consistency_loss,
    scale_shift_invariant_depth_loss,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.eval_utils import (
    average_metric_dicts,
    calculate_render_metrics,
    get_lpips_model,
    print_metric_block,
    write_tensorboard_metrics,
)
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, load_bts_geogs_config
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, psnr_max):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, optimization_args=opt)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    if opt.geometry_aware:
        if opt.depth_loss_enabled and not any(cam.depth_prior is not None for cam in scene.getTrainCameras()):
            print("[BTS-GeoGS] Depth loss requested but no depth prior was loaded; skipping it.")
        if opt.normal_loss_enabled and not any(cam.normal_prior is not None for cam in scene.getTrainCameras()):
            print("[BTS-GeoGS] Normal loss requested but no normal prior was loaded; skipping it.")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE, render_geometry=opt.geometry_aware)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss_terms = {"rgb": Ll1, "l1": Ll1, "ssim": 1.0 - ssim_value}
        geometry_weights = get_loss_weights(iteration, opt)
        confidence = viewpoint_cam.confidence_map
        if confidence is not None and opt.depth_confidence_weighted:
            confidence = torch.where(confidence >= opt.depth_min_confidence, confidence, torch.zeros_like(confidence))
        alpha_valid = render_pkg["alpha"] > 1e-6 if render_pkg["alpha"] is not None else None
        if geometry_weights["depth"] and viewpoint_cam.depth_prior is not None:
            depth_loss = scale_shift_invariant_depth_loss(render_pkg["depth"], viewpoint_cam.depth_prior,
                                                          confidence=confidence, valid_mask=alpha_valid)
            loss += geometry_weights["depth"] * depth_loss
            loss_terms["depth"] = depth_loss
        if geometry_weights["normal"] and viewpoint_cam.normal_prior is not None:
            normal_loss = normal_consistency_loss(render_pkg["normal"], viewpoint_cam.normal_prior,
                                                   confidence=confidence, valid_mask=alpha_valid,
                                                   use_abs_cosine=opt.normal_use_abs_cosine)
            loss += geometry_weights["normal"] * normal_loss
            loss_terms["normal"] = normal_loss
        edge_map = viewpoint_cam.edge_map
        if edge_map is not None and geometry_weights["edge"]:
            edge_loss = edge_weighted_l1_loss(image, gt_image, edge_map, opt.edge_weight_gamma)
            loss += geometry_weights["edge"] * edge_loss
            loss_terms["edge"] = edge_loss
        if geometry_weights["scale"]:
            scale_loss = gaussian_scale_regularization(gaussians.get_scaling, opt.max_gaussian_scale,
                                                       opt.max_anisotropy_ratio)
            loss += geometry_weights["scale"] * scale_loss
            loss_terms["scale"] = scale_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["invdepth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, psnr_max)
            if tb_writer:
                for name, value in loss_terms.items():
                    tb_writer.add_scalar(f"loss/{name}", value.item(), iteration)
                tb_writer.add_scalar("loss/total", loss.item(), iteration)
                tb_writer.add_scalar("gaussians/count", gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("gaussians/mean_opacity", gaussians.get_opacity.mean().item(), iteration)
                tb_writer.add_scalar("gaussians/mean_scale", gaussians.get_scaling.mean().item(), iteration)
                tb_writer.add_scalar("gaussians/max_scale", gaussians.get_scaling.max().item(), iteration)
                if opt.geometry_aware and iteration % 100 == 0:
                    tb_writer.add_images("render/depth", render_pkg["depth"].clamp_min(0)[None], iteration)
                    tb_writer.add_images("render/normal", ((render_pkg["normal"] + 1) * 0.5).clamp(0, 1)[None], iteration)
                    tb_writer.add_images("prior/edge", edge_map[None] if edge_map is not None else torch.zeros_like(render_pkg["depth"])[None], iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            gaussians.advance_gaussian_age()
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if opt.geometry_aware:
                    residual_map = (image.detach() - gt_image.detach()).abs().mean(dim=0)
                    if edge_map is None:
                        edge_map = torch.zeros_like(residual_map).unsqueeze(0)
                    gaussians.accumulate_geometry_stats(viewpoint_cam, visibility_filter, residual_map, edge_map, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if opt.geometry_aware:
                        gaussians.densify_and_prune_geometry(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, opt)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if (opt.geometry_aware and opt.importance_pruning_enabled and
                    iteration >= opt.importance_pruning_start_iter and
                    iteration % opt.importance_pruning_interval == 0):
                pruned = gaussians.prune_low_importance(opt)
                if pruned:
                    print(f"[BTS-GeoGS] Importance-pruned {pruned} Gaussians at iteration {iteration}.")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, psnr_max):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        lpips_model = get_lpips_model()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                metric_values = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if not getattr(viewpoint, "has_ground_truth", True):
                        continue
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    metric_values.append(calculate_render_metrics(image, gt_image, psnr_max, lpips_model))
                metrics = average_metric_dicts(metric_values)
                print_metric_block(iteration, f"{config['name']} ({scene.model_path})", metrics)
                if tb_writer and metrics is not None:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', metrics["l1"], iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', metrics["psnr"], iteration)
                    write_tensorboard_metrics(tb_writer, config['name'], iteration, metrics)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--psnr_max", type=float, default=30.0)
    parser.add_argument("--config", type=str, default=None, help="Optional BTS-GeoGS YAML preset; explicit CLI flags take precedence.")
    config_args, _ = parser.parse_known_args(sys.argv[1:])
    if config_args.config:
        parser.set_defaults(**load_bts_geogs_config(config_args.config))
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.psnr_max)

    # All done
    print("\nTraining complete.")
