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
import json
import time
import torch
from random import randint, choices
from utils.loss_utils import l1_loss, ssim
from utils.geometry_losses import (
    edge_weighted_l1_loss,
    gaussian_scale_regularization,
    get_loss_weights,
    normal_consistency_loss,
    ray_depth_variance_loss,
    scale_shift_invariant_depth_loss,
)
from utils.training_schedules import (
    get_resolution_stage,
    get_sh_degree,
    get_stage_loss_weights,
    piecewise_peak_weight,
    resolution_cache_scales,
)
from utils.depth_reprojection import pairwise_depth_consistency
from utils.densification_utils import corrected_residual_map
from utils.multiview_rgb import multiview_rgb_loss, warp_source_rgb_to_target
from utils.source_view_selection import select_source_views
from utils.patch_refinement import sample_patch
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.pose_refinement import PerViewPoseRefinement
from scene.blur_model import PerViewBlurTrajectory
from utils.blur_rendering import render_with_blur_formation
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.eval_utils import (
    average_metric_dicts,
    calculate_render_metrics,
    get_lpips_model,
    metrics_to_floats,
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
    scene = Scene(dataset, gaussians, resolution_scales=resolution_cache_scales(opt), optimization_args=opt)
    gaussians.training_setup(opt)
    training_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pose_model = None
    pose_optimizer = None
    if opt.pose_refinement_enabled:
        num_train_cameras = len(next(iter(scene.train_cameras.values())))
        pose_model = PerViewPoseRefinement(
            num_train_cameras, scene.cameras_extent,
            max_rotation_deg=opt.pose_max_rotation_deg,
            max_translation_radius_ratio=opt.pose_max_translation_radius_ratio,
        ).to(gaussians.get_xyz.device)
        pose_optimizer = torch.optim.Adam([
            {"params": [pose_model.rotation_raw], "lr": float(opt.pose_rotation_lr), "name": "rotation"},
            {"params": [pose_model.translation_raw], "lr": float(opt.pose_translation_lr), "name": "translation"},
        ], lr=0.0)
    blur_model = None
    blur_optimizer = None
    if opt.blur_formation_enabled:
        blur_model = PerViewBlurTrajectory.from_cameras(
            next(iter(scene.train_cameras.values())), scene.cameras_extent,
            percentile=opt.blur_sharpness_percentile_threshold,
            max_rotation_deg=opt.blur_max_rotation_deg,
            max_translation_radius_ratio=opt.blur_max_translation_radius_ratio,
        ).to(gaussians.get_xyz.device)
        blur_optimizer = torch.optim.Adam(blur_model.parameters(), lr=float(opt.pose_rotation_lr))
    if checkpoint:
        (checkpoint_payload, first_iter) = torch.load(checkpoint, weights_only=False)
        if isinstance(checkpoint_payload, dict) and "gaussians" in checkpoint_payload:
            gaussians.restore(checkpoint_payload["gaussians"], opt)
            if pose_model is not None and checkpoint_payload.get("pose_refinement") is not None:
                pose_model.load_state_dict(checkpoint_payload["pose_refinement"])
                if checkpoint_payload.get("pose_optimizer") is not None:
                    pose_optimizer.load_state_dict(checkpoint_payload["pose_optimizer"])
            elif pose_model is not None:
                print("[BTS-GeoGS] Checkpoint has no pose state; using identity train-pose corrections.")
            if blur_model is not None and checkpoint_payload.get("blur_trajectory") is not None:
                blur_model.load_state_dict(checkpoint_payload["blur_trajectory"])
                if checkpoint_payload.get("blur_optimizer") is not None:
                    blur_optimizer.load_state_dict(checkpoint_payload["blur_optimizer"])
            elif blur_model is not None:
                print("[BTS-GeoGS] Checkpoint has no blur state; using zero exposure trajectories.")
        else:
            gaussians.restore(checkpoint_payload, opt)
            if pose_model is not None:
                print("[BTS-GeoGS] Legacy checkpoint loaded; using identity train-pose corrections.")
            if blur_model is not None:
                print("[BTS-GeoGS] Legacy checkpoint loaded; using zero exposure trajectories.")

    perceptual_model = None
    if opt.perceptual_loss_enabled:
        perceptual_model = get_lpips_model()
        if perceptual_model is None:
            raise RuntimeError(
                "PERCEPTUAL.ENABLED requires the lpips package and AlexNet weights.")
        perceptual_model.requires_grad_(False)
        perceptual_model.eval()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    if opt.geometry_aware:
        diagnostic_cameras = next(iter(scene.train_cameras.values()))
        if opt.depth_loss_enabled and not any(cam.depth_prior is not None for cam in diagnostic_cameras):
            print("[BTS-GeoGS] Depth loss requested but no depth prior was loaded; skipping it.")
        if opt.normal_loss_enabled and not any(cam.normal_prior is not None for cam in diagnostic_cameras):
            print("[BTS-GeoGS] Normal loss requested but no normal prior was loaded; skipping it.")

    resolution_stage, image_scale = get_resolution_stage(first_iter, opt)
    resolution_key = 1.0 / image_scale
    viewpoint_stack = scene.getTrainCameras(resolution_key).copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    patch_center_history = {}

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
        pose_trainable = bool(
            pose_model is not None
            and int(opt.pose_refinement_start_iter) <= iteration <= int(opt.pose_refinement_freeze_iter))
        if pose_model is not None:
            for parameter in pose_model.parameters():
                parameter.requires_grad_(pose_trainable)
        if pose_optimizer is not None:
            for group in pose_optimizer.param_groups:
                group["lr"] = ((float(opt.pose_rotation_lr) if group["name"] == "rotation"
                                else float(opt.pose_translation_lr)) if pose_trainable else 0.0)
        blur_trainable = bool(
            blur_model is not None and int(opt.blur_start_iter) <= iteration <= int(opt.blur_freeze_iter))
        if blur_model is not None:
            for parameter in blur_model.parameters():
                parameter.requires_grad_(blur_trainable)
        if blur_optimizer is not None:
            for group in blur_optimizer.param_groups:
                group["lr"] = float(opt.pose_rotation_lr) if blur_trainable else 0.0

        scheduled_degree = get_sh_degree(iteration, opt, gaussians.max_sh_degree)
        if scheduled_degree is None:
            # Historic behavior remains exact when the schedule is disabled.
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
        else:
            gaussians.active_sh_degree = scheduled_degree

        next_stage, next_image_scale = get_resolution_stage(iteration, opt)
        if next_stage != resolution_stage:
            resolution_stage, image_scale = next_stage, next_image_scale
            resolution_key = 1.0 / image_scale
            viewpoint_stack = scene.getTrainCameras(resolution_key).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            gaussians.reset_image_space_statistics()
            print(f"[BTS-GeoGS] Resolution transition: stage={resolution_stage}, scale={image_scale:.3f}")
            if tb_writer:
                tb_writer.add_scalar("schedule/resolution_scale", image_scale, iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(resolution_key).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        if opt.sharpness_aware_sampling:
            sample_weights = [
                opt.sharpness_min_sample_weight
                + (opt.sharpness_max_sample_weight - opt.sharpness_min_sample_weight)
                * viewpoint_stack[index].normalized_sharpness
                for index in range(len(viewpoint_stack))
            ]
            rand_idx = choices(range(len(viewpoint_indices)), weights=sample_weights, k=1)[0]
        else:
            rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        training_camera = pose_model.refine_camera(viewpoint_cam) if pose_model is not None else viewpoint_cam

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        def render_training_camera(camera, model):
            return render(
                camera, model, pipe, bg, use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                render_geometry=(opt.geometry_aware or opt.multiview_depth_enabled
                                 or opt.multiview_rgb_enabled or opt.ray_depth_variance_enabled),
                apply_exposure=opt.exposure_compensation and iteration >= opt.exposure_start_iter,
                exposure_mode="training")

        if blur_model is not None and iteration >= int(opt.blur_start_iter):
            render_pkg = render_with_blur_formation(
                training_camera, gaussians, blur_model, render_training_camera, opt)
        else:
            render_pkg = render_training_camera(training_camera, gaussians)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        canonical_image = render_pkg["canonical_render"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image = image * alpha_mask
            canonical_image = canonical_image * alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        sky_mask = (viewpoint_cam.sky_mask.to(image.device, non_blocking=True)
                    if viewpoint_cam.sky_mask is not None else None)
        low_parallax_mask = (viewpoint_cam.low_parallax_mask.to(image.device, non_blocking=True)
                             if viewpoint_cam.low_parallax_mask is not None else None)
        edge_map = viewpoint_cam.edge_map
        if edge_map is not None:
            edge_map = edge_map.to(image.device, non_blocking=True)
        if edge_map is not None and opt.sky_enabled and sky_mask is not None:
            edge_map = edge_map * torch.where(
                sky_mask > 0.5, edge_map.new_tensor(float(opt.sky_edge_weight)), 1.0)
        if edge_map is not None and viewpoint_cam.local_sharpness is not None:
            local_sharpness = viewpoint_cam.local_sharpness.to(image.device, non_blocking=True)
            blur_weight = float(opt.blur_edge_weight_min) + (1.0 - float(opt.blur_edge_weight_min)) * local_sharpness
            edge_map = edge_map * blur_weight
        photometric_weight = torch.ones_like(gt_image[:1])
        if opt.sky_enabled and sky_mask is not None:
            photometric_weight = torch.where(
                sky_mask > 0.5,
                photometric_weight.new_tensor(float(opt.sky_photometric_weight)), photometric_weight)
        active_patch = None
        loss_image, loss_gt, loss_photometric = image, gt_image, photometric_weight
        loss_edge_map = edge_map
        if (opt.patch_refinement_enabled and iteration >= int(opt.patch_refinement_start_iter)
                and abs(float(image_scale) - 1.0) < 1e-8):
            patch_edge = edge_map if edge_map is not None else torch.zeros_like(gt_image[:1])
            thin_patch_map = getattr(viewpoint_cam, "thin_structure_mask", None)
            if thin_patch_map is not None:
                thin_patch_map = thin_patch_map.to(image.device, non_blocking=True)
            camera_patch_history = patch_center_history.setdefault(viewpoint_cam.image_name, [])
            active_patch = sample_patch(
                corrected_residual_map(image, gt_image, opt.densification_residual_type),
                patch_edge, int(opt.patch_refinement_patch_size),
                float(opt.patch_refinement_random_ratio),
                float(opt.patch_refinement_residual_ratio),
                float(opt.patch_refinement_edge_ratio),
                thin_structure_map=thin_patch_map,
                previous_centers=camera_patch_history,
                min_patch_distance=float(opt.patch_refinement_min_patch_distance))
            camera_patch_history.append((
                active_patch.top + 0.5 * (active_patch.height - 1),
                active_patch.left + 0.5 * (active_patch.width - 1)))
            del camera_patch_history[:-8]
            patch_slices = active_patch.slices
            loss_image, loss_gt = image[patch_slices], gt_image[patch_slices]
            loss_photometric = photometric_weight[patch_slices]
            if edge_map is not None:
                loss_edge_map = edge_map[patch_slices]
        Ll1 = l1_loss(loss_image, loss_gt)
        difference = loss_image - loss_gt
        if opt.use_charbonnier:
            reconstruction_map = torch.sqrt(difference.square() + float(opt.charbonnier_eps) ** 2).mean(dim=0, keepdim=True)
        else:
            reconstruction_map = difference.abs().mean(dim=0, keepdim=True)
        reconstruction_loss = (loss_photometric * reconstruction_map).sum() / loss_photometric.sum().clamp_min(1e-6)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(loss_image.unsqueeze(0), loss_gt.unsqueeze(0))
        else:
            ssim_value = ssim(loss_image, loss_gt)

        stage_weights = get_stage_loss_weights(iteration, opt)
        mse_value = torch.nn.functional.mse_loss(loss_image, loss_gt)
        loss = stage_weights["l1"] * reconstruction_loss + stage_weights["mse"] * mse_value + stage_weights["dssim"] * (1.0 - ssim_value)
        loss_terms = {"rgb": Ll1, "l1": Ll1, "mse": mse_value, "dssim": 1.0 - ssim_value}
        perceptual_active = (
            perceptual_model is not None
            and int(opt.perceptual_loss_start_iter) <= iteration <= int(opt.perceptual_loss_end_iter)
            and iteration % max(1, int(opt.perceptual_loss_interval)) == 0)
        if perceptual_active:
            perceptual_image = image.unsqueeze(0)
            perceptual_gt = gt_image.unsqueeze(0)
            max_size = max(32, int(opt.perceptual_loss_max_size))
            height, width = perceptual_image.shape[-2:]
            if max(height, width) > max_size:
                scale = max_size / float(max(height, width))
                target_size = (
                    max(32, round(height * scale)),
                    max(32, round(width * scale)))
                perceptual_image = torch.nn.functional.interpolate(
                    perceptual_image, size=target_size, mode="bilinear", align_corners=False)
                perceptual_gt = torch.nn.functional.interpolate(
                    perceptual_gt, size=target_size, mode="bilinear", align_corners=False)
            perceptual_loss = perceptual_model(
                perceptual_image * 2.0 - 1.0,
                perceptual_gt * 2.0 - 1.0).mean()
            loss = loss + float(opt.perceptual_loss_weight) * perceptual_loss
            loss_terms["perceptual"] = perceptual_loss
        geometry_weights = get_loss_weights(iteration, opt)
        confidence = (viewpoint_cam.confidence_map.to(image.device, non_blocking=True)
                      if viewpoint_cam.confidence_map is not None else None)
        if confidence is not None and opt.depth_confidence_weighted:
            confidence = torch.where(confidence >= opt.depth_min_confidence, confidence, torch.zeros_like(confidence))
        alpha_valid = render_pkg["alpha"] > 1e-6 if render_pkg["alpha"] is not None else None
        geometry_valid = alpha_valid
        if geometry_valid is not None and opt.sky_enabled and sky_mask is not None:
            geometry_valid = geometry_valid & (sky_mask < 0.5)
        geometry_confidence = confidence
        if opt.low_parallax_enabled and low_parallax_mask is not None:
            low_weight = torch.where(
                low_parallax_mask > 0.5,
                low_parallax_mask.new_tensor(float(opt.low_parallax_geometry_weight)),
                torch.ones_like(low_parallax_mask))
            geometry_confidence = low_weight if geometry_confidence is None else geometry_confidence * low_weight
        if geometry_weights["depth"] and viewpoint_cam.depth_prior is not None:
            depth_prior = viewpoint_cam.depth_prior.to(image.device, non_blocking=True)
            depth_loss = scale_shift_invariant_depth_loss(render_pkg["depth"], depth_prior,
                                                          confidence=geometry_confidence, valid_mask=geometry_valid)
            loss += stage_weights["geometry"] * geometry_weights["depth"] * depth_loss
            loss_terms["depth"] = depth_loss
        if geometry_weights["normal"] and viewpoint_cam.normal_prior is not None:
            normal_prior = viewpoint_cam.normal_prior.to(image.device, non_blocking=True)
            normal_loss = normal_consistency_loss(render_pkg["normal"], normal_prior,
                                                   confidence=geometry_confidence, valid_mask=geometry_valid,
                                                   use_abs_cosine=opt.normal_use_abs_cosine)
            loss += stage_weights["geometry"] * geometry_weights["normal"] * normal_loss
            loss_terms["normal"] = normal_loss
        if edge_map is not None and geometry_weights["edge"]:
            edge_loss = edge_weighted_l1_loss(
                loss_image, loss_gt, loss_edge_map, opt.edge_weight_gamma)
            loss += stage_weights["edge"] * geometry_weights["edge"] * edge_loss
            loss_terms["edge"] = edge_loss
        if geometry_weights["scale"]:
            scale_loss = gaussian_scale_regularization(gaussians.get_scaling, opt.max_gaussian_scale,
                                                       opt.max_anisotropy_ratio)
            loss += stage_weights["geometry"] * geometry_weights["scale"] * scale_loss
            loss_terms["scale"] = scale_loss
        ray_variance_weight = piecewise_peak_weight(
            iteration,
            int(opt.ray_depth_variance_start_iter),
            int(opt.ray_depth_variance_peak_iter),
            int(opt.ray_depth_variance_end_iter),
            float(opt.ray_depth_variance_weight_initial),
            float(opt.ray_depth_variance_weight_peak),
            float(opt.ray_depth_variance_weight_final),
        ) if opt.ray_depth_variance_enabled else 0.0
        if ray_variance_weight > 0.0:
            ray_valid = ((render_pkg["alpha"] >= float(opt.ray_depth_variance_min_alpha))
                         & (render_pkg["depth"] > 1e-6)
                         & torch.isfinite(render_pkg["depth"])
                         & torch.isfinite(render_pkg["depth_variance"]))
            ray_confidence = torch.ones_like(render_pkg["alpha"])
            uncertainty = render_pkg["uncertainty"]
            if uncertainty is not None:
                ray_valid &= uncertainty <= float(opt.ray_depth_variance_max_uncertainty)
                ray_confidence = ray_confidence * (1.0 - uncertainty.detach()).clamp(0.0, 1.0)
            if sky_mask is not None:
                ray_valid &= sky_mask < 0.5
            if edge_map is not None:
                edge_strength = edge_map.detach().clamp(0.0, 1.0)
                edge_multiplier = float(opt.ray_depth_variance_edge_weight_multiplier)
                ray_confidence = ray_confidence * (1.0 - edge_strength * (1.0 - edge_multiplier))
            thin_mask = getattr(viewpoint_cam, "thin_structure_mask", None)
            if thin_mask is not None:
                thin = thin_mask.to(image.device, non_blocking=True).clamp(0.0, 1.0)
                thin_multiplier = float(opt.ray_depth_variance_thin_weight_multiplier)
                ray_confidence = ray_confidence * (1.0 - thin * (1.0 - thin_multiplier))
            ray_variance = ray_depth_variance_loss(
                render_pkg["depth"], render_pkg["depth_variance"], render_pkg["alpha"],
                valid_mask=ray_valid, confidence=ray_confidence,
                min_alpha=float(opt.ray_depth_variance_min_alpha))
            ray_variance = torch.where(torch.isfinite(ray_variance), ray_variance,
                                       render_pkg["depth_variance"].sum() * 0.0)
            loss = loss + ray_variance_weight * ray_variance
            relative_variance = torch.nan_to_num(
                render_pkg["depth_variance"] / (render_pkg["depth"].square() + 1e-6),
                nan=0.0, posinf=0.0, neginf=0.0)
            loss_terms["ray_variance"] = ray_variance
            loss_terms["ray_variance_weight"] = image.new_tensor(ray_variance_weight)
            loss_terms["ray_variance_raw_mean"] = render_pkg["depth_variance"][ray_valid].mean() if ray_valid.any() else image.new_zeros(())
            loss_terms["ray_variance_relative_mean"] = relative_variance[ray_valid].mean() if ray_valid.any() else image.new_zeros(())
            loss_terms["ray_variance_valid_fraction"] = ray_valid.float().mean()
            if edge_map is not None:
                edge_pixels = ray_valid & (edge_map >= 0.5)
                non_edge_pixels = ray_valid & (edge_map < 0.5)
                loss_terms["ray_variance_edge_mean"] = (
                    relative_variance[edge_pixels].mean() if edge_pixels.any() else image.new_zeros(()))
                loss_terms["ray_variance_non_edge_mean"] = (
                    relative_variance[non_edge_pixels].mean() if non_edge_pixels.any() else image.new_zeros(()))
            if sky_mask is not None:
                finite_variance = torch.isfinite(relative_variance)
                sky_pixels = finite_variance & (sky_mask >= 0.5)
                non_sky_pixels = finite_variance & (sky_mask < 0.5)
                loss_terms["ray_variance_sky_mean"] = (
                    relative_variance[sky_pixels].mean() if sky_pixels.any() else image.new_zeros(()))
                loss_terms["ray_variance_non_sky_mean"] = (
                    relative_variance[non_sky_pixels].mean() if non_sky_pixels.any() else image.new_zeros(()))
        if opt.exposure_compensation and iteration >= opt.exposure_start_iter:
            exposure_loss = gaussians.exposure_regularization(
                opt.exposure_gain_reg_weight, opt.exposure_bias_reg_weight,
                opt.exposure_zero_mean_reg_weight)
            loss += stage_weights["exposure"] * exposure_loss
            loss_terms["exposure"] = exposure_loss
        if pose_trainable:
            pose_regularization = pose_model.regularization(
                opt.pose_rotation_reg_weight, opt.pose_translation_reg_weight)
            pose_smoothness = pose_model.trajectory_smoothness()
            loss = loss + pose_regularization + float(
                opt.pose_trajectory_smoothness_weight) * pose_smoothness
            loss_terms["pose_regularization"] = pose_regularization
            loss_terms["pose_smoothness"] = pose_smoothness
        if blur_trainable:
            blur_regularization = blur_model.regularization()
            loss = loss + float(opt.blur_trajectory_reg_weight) * blur_regularization
            loss_terms["blur_trajectory_regularization"] = blur_regularization

        depth_confidence_map = None
        selected_source_cameras = []
        source_packages = {}
        if (opt.multiview_depth_enabled and render_pkg["depth"] is not None
                and iteration % max(1, int(opt.multiview_depth_interval)) == 0
                and len(scene.getTrainCameras(resolution_key)) > 1):
            selected_source_cameras = select_source_views(
                viewpoint_cam, scene.getTrainCameras(resolution_key),
                max(1, int(opt.multiview_num_source_views)), scene.cameras_extent, opt)
            if selected_source_cameras:
                source_camera = selected_source_cameras[0]
                source_render_camera = (pose_model.refine_camera(source_camera)
                                        if pose_model is not None else source_camera)
                with torch.no_grad():
                    source_package = render(
                        source_render_camera, gaussians, pipe, bg, separate_sh=SPARSE_ADAM_AVAILABLE,
                        render_geometry=True, apply_exposure=False)
                source_packages[source_camera.image_name] = source_package
                consistency = pairwise_depth_consistency(
                    render_pkg["depth"], source_package["depth"], training_camera, source_render_camera,
                    target_alpha=render_pkg["alpha"], source_alpha=source_package["alpha"],
                    sky_mask=sky_mask, sigma_z=opt.multiview_depth_sigma,
                    relative_threshold=opt.multiview_depth_relative_threshold)
                depth_confidence_map = consistency.soft_confidence.detach()
                detached_confidence = consistency.soft_confidence.detach()
                valid_confidence = detached_confidence.sum()
                if valid_confidence > 0:
                    multiview_depth_loss = (
                        consistency.relative_error * detached_confidence).sum() / valid_confidence.clamp_min(1e-6)
                    loss = loss + float(opt.multiview_depth_weight) * multiview_depth_loss
                    loss_terms["multiview_depth"] = multiview_depth_loss

        multiview_weight = piecewise_peak_weight(
            iteration, int(opt.multiview_start_iter), int(opt.multiview_peak_iter),
            int(opt.multiview_end_iter), float(opt.multiview_weight_initial),
            float(opt.multiview_weight_peak), float(opt.multiview_weight_final),
        ) if opt.multiview_rgb_enabled else 0.0
        if (multiview_weight > 0.0 and render_pkg["depth"] is not None
                and iteration % max(1, int(opt.multiview_interval)) == 0
                and len(scene.getTrainCameras(resolution_key)) > 1):
            if not selected_source_cameras:
                selected_source_cameras = select_source_views(
                    viewpoint_cam, scene.getTrainCameras(resolution_key),
                    int(opt.multiview_num_source_views), scene.cameras_extent, opt)
            source_losses = []
            source_valid_fractions = []
            source_zncc = []
            source_gradient = []
            for source_camera in selected_source_cameras:
                source_render_camera = (pose_model.refine_camera(source_camera)
                                        if pose_model is not None else source_camera)
                source_package = source_packages.get(source_camera.image_name)
                if source_package is None:
                    with torch.no_grad():
                        source_package = render(
                            source_render_camera, gaussians, pipe, bg, separate_sh=SPARSE_ADAM_AVAILABLE,
                            render_geometry=True, apply_exposure=False)
                    source_packages[source_camera.image_name] = source_package
                warped = warp_source_rgb_to_target(
                    render_pkg["depth"], training_camera,
                    source_camera.original_image.to(image.device, non_blocking=True),
                    source_package["depth"].detach(), source_render_camera,
                    render_pkg["alpha"], source_package["alpha"].detach(), sky_mask, opt)
                components = multiview_rgb_loss(
                    image, warped["warped_rgb"], warped["valid_mask"],
                    warped["depth_confidence"].detach(), opt)
                if int(warped["valid_mask"].sum().item()) >= int(opt.multiview_min_valid_pixels):
                    source_losses.append(components["total"])
                    source_valid_fractions.append(components["valid_fraction"])
                    source_zncc.append(components["zncc"])
                    source_gradient.append(components["gradient"])
            if source_losses:
                multiview_photo = torch.stack(source_losses).mean()
                multiview_photo = torch.where(
                    torch.isfinite(multiview_photo), multiview_photo, image.sum() * 0.0)
                loss = loss + multiview_weight * multiview_photo
                loss_terms["multiview_rgb"] = multiview_photo
                loss_terms["multiview_rgb_weight"] = image.new_tensor(multiview_weight)
                loss_terms["multiview_valid_fraction"] = torch.stack(source_valid_fractions).mean()
                loss_terms["multiview_zncc"] = torch.stack(source_zncc).mean()
                loss_terms["multiview_gradient"] = torch.stack(source_gradient).mean()
                if iteration % 500 == 0:
                    print("[BTS-GeoGS] Multi-view sources for " + viewpoint_cam.image_name + ": "
                          + ", ".join(camera.image_name for camera in selected_source_cameras))

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

        if not torch.isfinite(loss):
            diagnostics = {
                name: (float(value.detach().item()) if value.numel() == 1 else "non-scalar")
                for name, value in loss_terms.items()
            }
            raise FloatingPointError(
                f"Non-finite Stage-1 loss at iteration {iteration}: {diagnostics}")
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
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, pipe, background, SPARSE_ADAM_AVAILABLE,
                            dataset.train_test_exp, psnr_max, opt, pose_model, blur_model)
            if tb_writer:
                for name, value in loss_terms.items():
                    tb_writer.add_scalar(f"loss/{name}", value.item(), iteration)
                tb_writer.add_scalar("loss/total", loss.item(), iteration)
                tb_writer.add_scalar("gaussians/count", gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("gaussians/mean_opacity", gaussians.get_opacity.mean().item(), iteration)
                tb_writer.add_scalar("gaussians/mean_scale", gaussians.get_scaling.mean().item(), iteration)
                tb_writer.add_scalar("gaussians/max_scale", gaussians.get_scaling.max().item(), iteration)
                tb_writer.add_scalar("schedule/active_sh_degree", gaussians.active_sh_degree, iteration)
                tb_writer.add_scalar("schedule/resolution_scale", image_scale, iteration)
                tb_writer.add_scalar("image_quality/sharpness", viewpoint_cam.sharpness, iteration)
                if active_patch is not None:
                    patch_category = {"random": 0, "residual": 1, "edge": 2}[active_patch.category]
                    tb_writer.add_scalar("patch/category", patch_category, iteration)
                    tb_writer.add_scalar("patch/left", active_patch.left, iteration)
                    tb_writer.add_scalar("patch/top", active_patch.top, iteration)
                if gaussians.visibility_count.numel():
                    tb_writer.add_scalar("gaussians/visibility_mean", gaussians.visibility_count.float().mean().item(), iteration)
                if opt.exposure_compensation and gaussians.exposure_model is not None:
                    if gaussians.exposure_spline is not None:
                        gains, biases = gaussians.exposure_spline.gains_biases()
                    else:
                        gains, biases = gaussians.exposure_model.gains(), gaussians.exposure_model.biases()
                    for channel, name in enumerate(("r", "g", "b")):
                        tb_writer.add_scalar(f"exposure/gain_mean_{name}", gains[:, channel].mean().item(), iteration)
                    tb_writer.add_scalar("exposure/gain_min", gains.min().item(), iteration)
                    tb_writer.add_scalar("exposure/gain_max", gains.max().item(), iteration)
                    tb_writer.add_scalar("exposure/bias_mean", biases.mean().item(), iteration)
                    tb_writer.add_scalar("exposure/bias_abs_max", biases.abs().max().item(), iteration)
                    tb_writer.add_scalar("exposure/regularization", loss_terms.get("exposure", image.new_zeros(())).item(), iteration)
                    tb_writer.add_scalar("exposure/out_of_range_fraction", gaussians.exposure_out_of_range_fraction(image).item(), iteration)
                    if iteration % 100 == 0:
                        tb_writer.add_images("render/canonical", canonical_image.clamp(0, 1)[None], iteration)
                        tb_writer.add_images("render/corrected", image.clamp(0, 1)[None], iteration)
                        tb_writer.add_images("gt", gt_image[None], iteration)
                        tb_writer.add_images("error/canonical", (canonical_image.detach() - gt_image).abs().clamp(0, 1)[None], iteration)
                        tb_writer.add_images("error/corrected", (image.detach() - gt_image).abs().clamp(0, 1)[None], iteration)
                if pose_model is not None:
                    for name, value in pose_model.diagnostics(scene.cameras_extent).items():
                        tb_writer.add_scalar(f"pose/{name}", value, iteration)
                if blur_model is not None:
                    tb_writer.add_scalar("blur/blurred_view_fraction",
                                         blur_model.blurred_mask.float().mean().item(), iteration)
                    tb_writer.add_scalar("blur/current_view_is_blurred",
                                         float(blur_model.is_blurred(viewpoint_cam.view_index)), iteration)
                if opt.geometry_aware and iteration % 100 == 0:
                    tb_writer.add_images("render/depth", render_pkg["depth"].clamp_min(0)[None], iteration)
                    tb_writer.add_images("render/normal", ((render_pkg["normal"] + 1) * 0.5).clamp(0, 1)[None], iteration)
                    tb_writer.add_images("render/alpha", render_pkg["alpha"].clamp(0, 1)[None], iteration)
                    tb_writer.add_images("render/uncertainty", render_pkg["uncertainty"].clamp(0, 1)[None], iteration)
                    if sky_mask is not None:
                        tb_writer.add_images("prior/sky", sky_mask[None], iteration)
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
                if gaussians.densification_stats_enabled:
                    gradient_only = (
                        opt.densification_method
                        in {"absolute_gradient", "absolute_gradient_approx", "rms_gradient"}
                        and not opt.densification_residual_aware
                        and not opt.densification_edge_aware
                        and not opt.footprint_sampling_enabled
                        and not opt.spatial_budget_enabled
                    )
                    if gradient_only:
                        gaussians.accumulate_visibility_only(visibility_filter)
                    else:
                        residual_map = corrected_residual_map(image, gt_image, opt.densification_residual_type)
                        if edge_map is None:
                            edge_map = torch.zeros_like(residual_map)
                        if (blur_model is not None and blur_model.is_blurred(viewpoint_cam.view_index)
                                and iteration <= int(opt.blur_freeze_iter)):
                            residual_map = residual_map * float(opt.blur_densification_weight)
                            edge_map = edge_map * float(opt.blur_formation_edge_weight)
                        gaussians.accumulate_geometry_stats(
                            training_camera, visibility_filter, residual_map, edge_map, radii, depth_confidence_map)

                densification_window = (int(opt.densification_window_size)
                                        if opt.densification_method == "persistent_multiview_hybrid"
                                        else int(opt.densification_interval))
                if iteration > opt.densify_from_iter and iteration % max(1, densification_window) == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if gaussians.densification_stats_enabled:
                        gaussians.densify_and_prune_geometry(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                            size_threshold, radii, opt, iteration)
                        if tb_writer:
                            for name, value in gaussians._last_densification_metrics.items():
                                tb_writer.add_scalar(f"densification/{name}", value, iteration)
                            for name, value in gaussians._last_densify_counts.items():
                                tb_writer.add_scalar(f"densification/{name}_count", value, iteration)
                            if (opt.densification_method == "persistent_multiview_hybrid"
                                    and gaussians.gradient_burstiness.numel()):
                                tb_writer.add_histogram("densification/burstiness", gaussians.gradient_burstiness, iteration)
                                tb_writer.add_histogram("densification/unique_view_support", gaussians.unique_view_support.float(), iteration)
                                tb_writer.add_histogram("densification/persistent_hits", gaussians.window_hit_count.float(), iteration)
                            if gaussians._last_spatial_selection is not None:
                                spatial = gaussians._last_spatial_selection
                                energy = spatial.tile_energy
                                normalized_energy = energy / energy.amax().clamp_min(1e-6)
                                budget_map = spatial.tile_budget.float()
                                normalized_budget = budget_map / budget_map.amax().clamp_min(1.0)
                                tb_writer.add_images("densification/tile_error_heatmap", normalized_energy[None, None], iteration)
                                tb_writer.add_images("densification/tile_budget", normalized_budget[None, None], iteration)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if (opt.geometry_aware and opt.importance_pruning_enabled and
                    iteration >= opt.importance_pruning_start_iter and
                    iteration % opt.importance_pruning_interval == 0):
                pruned = gaussians.prune_low_importance(opt, iteration)
                if pruned:
                    print(f"[BTS-GeoGS] Importance-pruned {pruned} Gaussians at iteration {iteration}.")

            # Optimizer step
            if iteration < opt.iterations:
                if gaussians.background_optimizer is not None:
                    gaussians.background_optimizer.step()
                    gaussians.background_optimizer.zero_grad(set_to_none=True)
                if (opt.exposure_compensation and opt.exposure_start_iter <= iteration <= opt.exposure_end_iter
                        and iteration <= opt.exposure_freeze_iter):
                    gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if pose_optimizer is not None:
                    if pose_trainable:
                        pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
                if blur_optimizer is not None:
                    if blur_trainable:
                        blur_optimizer.step()
                    blur_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_payload = gaussians.capture()
                if pose_model is not None or blur_model is not None:
                    checkpoint_payload = {
                        "gaussians": checkpoint_payload,
                        "pose_refinement": pose_model.state_dict() if pose_model is not None else None,
                        "pose_optimizer": pose_optimizer.state_dict() if pose_optimizer is not None else None,
                        "blur_trajectory": blur_model.state_dict() if blur_model is not None else None,
                        "blur_optimizer": blur_optimizer.state_dict() if blur_optimizer is not None else None,
                    }
                torch.save((checkpoint_payload, iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    runtime = {
        "train_time_seconds": time.perf_counter() - training_started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "final_iteration": int(opt.iterations),
    }
    with open(os.path.join(scene.model_path, "stage1_runtime.json"), "w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2)

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
    config_path = getattr(args, "resolved_config_source", None)
    if config_path:
        from utils.config_utils import load_yaml_with_base
        import yaml
        with open(os.path.join(args.model_path, "resolved_config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(load_yaml_with_base(config_path), handle, sort_keys=False)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene : Scene,
                    pipe, background, separate_sh, train_test_exp, psnr_max, opt,
                    pose_model=None, blur_model=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        lpips_model = get_lpips_model()
        validation_record = {
            "iteration": int(iteration),
            "gaussian_count": int(scene.gaussians.get_xyz.shape[0]),
            "splits": {},
        }
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                split_record = {"summary": None, "canonical_summary": None,
                                "per_view": [], "difficulty": {}}
                metric_values = []
                canonical_metric_values = []
                metrics_by_difficulty = {}
                for idx, viewpoint in enumerate(config['cameras']):
                    is_train = config['name'] == 'train'
                    render_viewpoint = (pose_model.refine_camera(viewpoint)
                                        if is_train and pose_model is not None else viewpoint)
                    use_v3_exposure = bool(opt.exposure_compensation)
                    exposure_mode = "training" if is_train else opt.test_exposure_mode
                    def render_validation_camera(camera, model):
                        return render(
                            camera, model, pipe, background,
                            use_trained_exp=train_test_exp, separate_sh=separate_sh,
                            apply_exposure=use_v3_exposure, exposure_mode=exposure_mode)
                    if is_train and blur_model is not None:
                        package = render_with_blur_formation(
                            render_viewpoint, scene.gaussians, blur_model,
                            render_validation_camera, opt)
                    else:
                        package = render_validation_camera(render_viewpoint, scene.gaussians)
                    if use_v3_exposure and not is_train and scene.gaussians.last_exposure_diagnostics:
                        diagnostics = scene.gaussians.last_exposure_diagnostics
                        source_names = {index: name for name, index in scene.gaussians.exposure_mapping.items()}
                        selected = [source_names.get(index, str(index)) for index in diagnostics.get("indices", [])]
                        print(f"[Exposure] target={viewpoint.image_name} sources={selected} "
                              f"weights={diagnostics.get('weights', [])} "
                              f"nearest={diagnostics.get('nearest_distance', 0.0):.5f} "
                              f"confidence={diagnostics.get('confidence', 0.0):.5f} "
                              f"out_of_range={diagnostics.get('out_of_range_fraction', 0.0):.6f}")
                    canonical_image = torch.clamp(package["canonical_render"], 0.0, 1.0)
                    image = torch.clamp(package["corrected_render"] if use_v3_exposure else package["render"], 0.0, 1.0)
                    if not getattr(viewpoint, "has_ground_truth", True):
                        continue
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        canonical_image = canonical_image[..., canonical_image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    view_metrics = calculate_render_metrics(image, gt_image, psnr_max, lpips_model)
                    metric_values.append(view_metrics)
                    difficulty = getattr(viewpoint, "difficulty_bin", "")
                    split_record["per_view"].append({
                        "image_name": viewpoint.image_name,
                        "difficulty": difficulty or None,
                        **metrics_to_floats(view_metrics),
                    })
                    if difficulty:
                        metrics_by_difficulty.setdefault(difficulty, []).append(view_metrics)
                    if is_train:
                        canonical_metric_values.append(calculate_render_metrics(
                            canonical_image, gt_image, psnr_max, lpips_model))
                metrics = average_metric_dicts(metric_values)
                split_record["summary"] = metrics_to_floats(metrics)
                print_metric_block(iteration, f"{config['name']} ({scene.model_path})", metrics)
                canonical_metrics = average_metric_dicts(canonical_metric_values)
                split_record["canonical_summary"] = metrics_to_floats(canonical_metrics)
                if canonical_metrics is not None:
                    print_metric_block(iteration, f"{config['name']} canonical ({scene.model_path})", canonical_metrics)
                if tb_writer and metrics is not None:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', metrics["l1"], iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', metrics["psnr"], iteration)
                    write_tensorboard_metrics(tb_writer, config['name'], iteration, metrics)
                    millions = max(scene.gaussians.get_xyz.shape[0] / 1_000_000.0, 1e-6)
                    tb_writer.add_scalar(config['name'] + '/quality_psnr_per_million_gaussians',
                                         metrics["psnr"].item() / millions, iteration)
                    if config['name'] == 'train':
                        tb_writer.add_scalar("eval/train_corrected_psnr", metrics["psnr"].item(), iteration)
                        tb_writer.add_scalar("eval/train_corrected_ssim", metrics["ssim"].item(), iteration)
                for difficulty, values in sorted(metrics_by_difficulty.items()):
                    difficulty_metrics = average_metric_dicts(values)
                    split_record["difficulty"][difficulty] = metrics_to_floats(difficulty_metrics)
                    print_metric_block(iteration, f"{config['name']}/{difficulty} ({scene.model_path})",
                                       difficulty_metrics)
                    if tb_writer and difficulty_metrics is not None:
                        write_tensorboard_metrics(tb_writer, f"{config['name']}/difficulty/{difficulty}",
                                                  iteration, difficulty_metrics)
                if tb_writer and canonical_metrics is not None:
                    tb_writer.add_scalar("eval/train_canonical_psnr", canonical_metrics["psnr"].item(), iteration)
                    tb_writer.add_scalar("eval/train_canonical_ssim", canonical_metrics["ssim"].item(), iteration)
                validation_record["splits"][config["name"]] = split_record

        with open(os.path.join(scene.model_path, "stage1_validation_metrics.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(validation_record, handle, indent=2)

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
    parser.add_argument("--seed", type=int, default=0)
    config_args, _ = parser.parse_known_args(sys.argv[1:])
    if config_args.config:
        parser.set_defaults(**load_bts_geogs_config(config_args.config))
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    dataset_args.resolved_config_source = args.config
    training(dataset_args, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.psnr_max)

    # All done
    print("\nTraining complete.")
