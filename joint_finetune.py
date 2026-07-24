"""Optional topology-locked joint fine-tuning for BTS GeoNAF-GS Phase 3."""

from __future__ import annotations

import argparse
import json
import os
import random
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path

import torch

from arguments import OptimizationParams
from utils.loss_utils import ssim
from utils.stage2_gaussian import load_gaussian_scene, serialize_camera
from utils.stage2_geometry import prepare_stage2_input_from_render
from utils.stage2_io import load_refiner_checkpoint, load_stage2_config
from utils.stage2_losses import Stage2Loss
from utils.stage2_multiview import forward_warp_rgb, masked_multiview_l1

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _default_optimization_args():
    parser = ArgumentParser(add_help=False)
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args([]))


def _camera_tensor(camera, index: int, device: torch.device):
    metadata = serialize_camera(camera, index)
    intrinsics = metadata["intrinsics"]
    return (
        torch.tensor(
            [
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["cx"],
                intrinsics["cy"],
            ],
            device=device,
        ),
        torch.tensor(metadata["extrinsics"], device=device),
    )


def _nearest_pairs(cameras) -> list[int]:
    centers = torch.stack([camera.camera_center.detach().cpu() for camera in cameras])
    directions = torch.tensor(
        [camera.R[:, 2].tolist() for camera in cameras], dtype=torch.float32
    )
    directions = torch.nn.functional.normalize(directions, dim=1)
    distances = torch.cdist(centers, centers)
    positive = distances[distances > 0]
    scale = positive.median() if positive.numel() else distances.new_tensor(1.0)
    scores = distances / scale.clamp_min(1e-6) + 1.0 - directions @ directions.t()
    scores.fill_diagonal_(float("inf"))
    return [int(row.argmin().item()) for row in scores]


def _grad_norm(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().norm().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional topology-locked BTS GeoNAF-GS joint fine-tuning"
    )
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--refiner_checkpoint", required=True)
    parser.add_argument(
        "--config", default="configs/stage2/geonaf_base.yaml"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--enable_joint",
        action="store_true",
        help="Explicit acknowledgement that Phase 3 is experimental.",
    )
    args = parser.parse_args()

    config = load_stage2_config(args.config)
    joint_config = config.get("JOINT", {})
    if not args.enable_joint and not bool(joint_config.get("ENABLED", False)):
        raise RuntimeError(
            "Joint fine-tuning is disabled by default. Pass --enable_joint or "
            "set JOINT.ENABLED=true after completing independent Stage 1/2."
        )
    device = torch.device(args.device)
    (
        dataset,
        pipeline,
        optimization,
        gaussians,
        scene,
        background,
    ) = load_gaussian_scene(
        args.source_path,
        args.model_path,
        args.iteration,
        resolution=args.resolution,
        device=args.device,
    )
    optimization = optimization or _default_optimization_args()
    # A training-time initialization cap is not a valid Phase-3 operation.
    optimization.initialization_cap_to_max_gaussians = False
    count_before_setup = int(gaussians.get_xyz.shape[0])
    # No pose/blur model is created, and exposure/background optimizers are
    # intentionally never stepped.
    gaussians.training_setup(optimization)
    if int(gaussians.get_xyz.shape[0]) != count_before_setup:
        raise AssertionError("Gaussian topology changed while setting up Phase 3")
    gaussian_lr_scale = float(joint_config.get("GAUSSIAN_LR_SCALE", 0.01))
    for group in gaussians.optimizer.param_groups:
        group["lr"] *= gaussian_lr_scale

    refiner, refiner_payload = load_refiner_checkpoint(
        args.refiner_checkpoint, config, device
    )
    refiner.train()
    refiner_optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=float(joint_config.get("NAF_LR", 2e-5)),
        weight_decay=float(config["TRAIN"].get("WEIGHT_DECAY", 1e-4)),
    )
    criterion = Stage2Loss(config["LOSS"]).to(device)
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError("No training camera is available for joint fine-tuning")
    pairs = _nearest_pairs(cameras) if len(cameras) > 1 else [0]
    steps = int(
        args.steps if args.steps is not None else joint_config.get("STEPS", 1000)
    )
    lambda_final = float(joint_config.get("LAMBDA_FINAL", 1.0))
    lambda_multiview = float(joint_config.get("LAMBDA_MULTIVIEW", 0.0))
    direct_l1_weight = float(joint_config.get("GS_L1", 0.8))
    direct_mse_weight = float(joint_config.get("GS_MSE", 0.0))
    direct_dssim_weight = float(joint_config.get("GS_DSSIM", 0.2))
    initial_count = int(gaussians.get_xyz.shape[0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_dir / "tensorboard")) if SummaryWriter else None
    amp_enabled = bool(config["TRAIN"].get("AMP", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    from gaussian_renderer import render

    for step in range(1, steps + 1):
        camera_index = random.randrange(len(cameras))
        camera = cameras[camera_index]
        gaussians.optimizer.zero_grad(set_to_none=True)
        refiner_optimizer.zero_grad(set_to_none=True)
        package = render(
            camera,
            gaussians,
            pipeline,
            background,
            render_geometry=True,
            apply_exposure=False,
        )
        gaussian_rgb = package["render"].clamp(0.0, 1.0)
        target = camera.original_image[:3].to(device)
        if camera.alpha_mask is not None:
            alpha_mask = camera.alpha_mask.to(device)
            gaussian_rgb = gaussian_rgb * alpha_mask
            target = target * alpha_mask
        stage2_package = dict(package)
        stage2_package["render"] = gaussian_rgb
        stage2_input, _ = prepare_stage2_input_from_render(stage2_package, config)
        neighbor_package = None
        neighbor_input = None
        neighbor_camera = None
        neighbor_index = None
        if lambda_multiview > 0.0 and len(cameras) > 1:
            neighbor_index = pairs[camera_index]
            neighbor_camera = cameras[neighbor_index]
            # Keep the custom CUDA rasterizer outside the AMP region.
            neighbor_package = render(
                neighbor_camera,
                gaussians,
                pipeline,
                background,
                render_geometry=True,
                apply_exposure=False,
            )
            neighbor_input, _ = prepare_stage2_input_from_render(
                neighbor_package, config
            )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            output = refiner(
                stage2_input,
                gaussian_rgb=gaussian_rgb.unsqueeze(0),
                uncertainty=package["uncertainty"].unsqueeze(0),
            )
            difference = gaussian_rgb - target
            direct = (
                direct_l1_weight * difference.abs().mean()
                + direct_mse_weight * difference.square().mean()
                + direct_dssim_weight
                * (1.0 - ssim(gaussian_rgb.unsqueeze(0), target.unsqueeze(0)))
            )
            naf_terms = criterion(
                output, gaussian_rgb.unsqueeze(0), target.unsqueeze(0)
            )
            multiview = direct.new_zeros(())
            if neighbor_package is not None:
                assert neighbor_input is not None
                assert neighbor_camera is not None
                assert neighbor_index is not None
                neighbor_output = refiner(
                    neighbor_input,
                    gaussian_rgb=neighbor_package["render"].unsqueeze(0),
                    uncertainty=neighbor_package["uncertainty"].unsqueeze(0),
                )
                source_intrinsics, source_extrinsics = _camera_tensor(
                    camera, camera_index, device
                )
                target_intrinsics, target_extrinsics = _camera_tensor(
                    neighbor_camera, neighbor_index, device
                )
                warped, valid = forward_warp_rgb(
                    output["final_rgb"][0],
                    package["depth"],
                    neighbor_package["depth"],
                    source_intrinsics,
                    target_intrinsics,
                    source_extrinsics,
                    target_extrinsics,
                    source_alpha=package["alpha"],
                    target_alpha=neighbor_package["alpha"],
                    source_uncertainty=package["uncertainty"],
                    target_uncertainty=neighbor_package["uncertainty"],
                )
                multiview = masked_multiview_l1(
                    warped, neighbor_output["final_rgb"][0], valid
                )
            total = direct + lambda_final * naf_terms["total"]
            total = total + lambda_multiview * multiview
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite joint loss at step {step}")
        scaler.scale(total).backward()
        scaler.unscale_(gaussians.optimizer)
        scaler.unscale_(refiner_optimizer)
        naf_grad = torch.nn.utils.clip_grad_norm_(
            refiner.parameters(), float(config["TRAIN"].get("GRAD_CLIP", 1.0))
        )
        xyz_grad = _grad_norm(gaussians._xyz.grad)
        sh_grad = (
            _grad_norm(gaussians._features_dc.grad) ** 2
            + _grad_norm(gaussians._features_rest.grad) ** 2
        ) ** 0.5
        scaler.step(gaussians.optimizer)
        scaler.step(refiner_optimizer)
        scaler.update()
        if int(gaussians.get_xyz.shape[0]) != initial_count:
            raise AssertionError(
                "Gaussian topology changed during topology-locked fine-tuning"
            )
        if writer is not None:
            writer.add_scalar("loss/total", float(total.detach()), step)
            writer.add_scalar("loss/gs_direct", float(direct.detach()), step)
            writer.add_scalar("loss/naf", float(naf_terms["total"].detach()), step)
            writer.add_scalar("loss/multiview", float(multiview.detach()), step)
            writer.add_scalar("grad/gaussian_position", xyz_grad, step)
            writer.add_scalar("grad/gaussian_sh", sh_grad, step)
            writer.add_scalar("grad/nafnet", float(naf_grad), step)
        if step % 50 == 0 or step == steps:
            print(
                f"Joint step {step}/{steps}: total={float(total.detach()):.6f}, "
                f"GS={float(direct.detach()):.6f}, "
                f"NAF={float(naf_terms['total'].detach()):.6f}"
            )

    gaussian_path = output_dir / "point_cloud" / "point_cloud.ply"
    gaussians.save_ply(str(gaussian_path))
    payload = {
        "gaussians": gaussians.capture(),
        "refiner": refiner.state_dict(),
        "gaussian_optimizer": gaussians.optimizer.state_dict(),
        "refiner_optimizer": refiner_optimizer.state_dict(),
        "source_gaussian_iteration": int(scene.loaded_iter),
        "source_refiner_epoch": refiner_payload.get("epoch"),
        "steps": steps,
        "config": config,
        "topology_locked": True,
    }
    temporary = output_dir / "joint_latest.pth.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output_dir / "joint_latest.pth")
    (output_dir / "joint_summary.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "gaussian_count": initial_count,
                "densification": False,
                "clone_split_prune": False,
                "pose_refinement": False,
                "blur_model": False,
                "exposure_frozen": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if writer is not None:
        writer.close()
    print(f"Saved topology-locked joint checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
