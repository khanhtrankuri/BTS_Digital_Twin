"""Render AbsGS test cameras and apply the frozen GeoNAF residual refiner."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.stage2_gaussian import load_gaussian_scene
from utils.stage2_geometry import prepare_stage2_input_from_render
from utils.stage2_inference import tiled_refine
from utils.stage2_io import (
    load_refiner_checkpoint,
    load_stage2_config,
    save_tensor_image,
)


def _redistortion_metadata(scene_path: Path) -> dict:
    metadata_path = scene_path / "undistortion_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Redistortion requested but metadata is missing: {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    cameras = payload.get("cameras", {})
    if len(cameras) != 1:
        raise ValueError(
            "Submission redistortion requires one shared camera model per scene"
        )
    return next(iter(cameras.values()))


def _redistort_tensor(value: torch.Tensor, metadata: dict) -> torch.Tensor:
    from tools.prepare_undistorted_scene import redistort_render

    value = value.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    array = (
        value.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    result = redistort_render(array, metadata)
    if result.ndim == 2:
        result = result[..., None]
    return torch.from_numpy(np.asarray(result).copy()).permute(2, 0, 1).float() / 255.0


def _output_name(image_name: str) -> str:
    path = Path(image_name)
    return path.name if path.suffix else f"{path.name}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render BTS GeoNAF-GS raw and refined outputs"
    )
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--refiner_checkpoint", required=True)
    parser.add_argument(
        "--refiner_config", default="configs/stage2/geonaf_base.yaml"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument(
        "--split", choices=("train", "test"), default="test"
    )
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--tile_overlap", type=int, default=None)
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--exposure_compensation", action="store_true")
    parser.add_argument(
        "--test_exposure_mode",
        choices=(
            "identity",
            "nearest_camera",
            "weighted_nearest",
            "pose_confidence_blend",
            "temporal_weighted",
            "pose_temporal_weighted",
            "temporal_spline",
        ),
        default="identity",
    )
    parser.add_argument("--redistort_to_source_grid", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_stage2_config(args.refiner_config)
    model, _ = load_refiner_checkpoint(
        args.refiner_checkpoint, config, device
    )
    model.eval()
    dataset, pipeline, _, gaussians, scene, background = load_gaussian_scene(
        args.source_path,
        args.model_path,
        args.iteration,
        resolution=args.resolution,
        device=args.device,
    )
    cameras = (
        scene.getTrainCameras()
        if args.split == "train"
        else scene.getTestCameras()
    )
    if not cameras:
        raise ValueError(f"No {args.split} cameras were loaded")
    output_root = Path(args.output_dir)
    directories = {
        name: output_root / name
        for name in ("raw", "refined", "mask", "residual", "uncertainty")
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    inference_config = config["INFERENCE"]
    tile_size = (
        int(args.tile_size)
        if args.tile_size is not None
        else int(inference_config.get("TILE_SIZE", 512))
    )
    overlap = (
        int(args.tile_overlap)
        if args.tile_overlap is not None
        else int(inference_config.get("TILE_OVERLAP", 64))
    )
    save_mask = args.save_mask or bool(inference_config.get("SAVE_MASK", False))
    redistortion = (
        _redistortion_metadata(Path(args.source_path))
        if args.redistort_to_source_grid
        else None
    )
    amp_enabled = device.type == "cuda"

    from gaussian_renderer import render

    with torch.no_grad():
        for camera in tqdm(cameras, desc=f"GeoNAF render ({args.split})"):
            package = render(
                camera,
                gaussians,
                pipeline,
                background,
                render_geometry=True,
                apply_exposure=args.exposure_compensation,
                exposure_mode=(
                    "training"
                    if args.split == "train"
                    else args.test_exposure_mode
                ),
            )
            stage2_input, _ = prepare_stage2_input_from_render(package, config)
            stage2_input = stage2_input.to(device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with amp_context:
                output = tiled_refine(
                    model,
                    stage2_input,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            raw = package["render"].float().clamp(0.0, 1.0)
            refined = output["final_rgb"][0].float()
            mask = output["effective_mask"][0].float()
            correction = output["correction"][0].float()
            residual_scale = float(config["MODEL"]["RESIDUAL_SCALE"])
            residual_visualization = (
                0.5 + correction / (2.0 * residual_scale)
            ).clamp(0.0, 1.0)
            uncertainty = package["uncertainty"].float()
            products = {
                "raw": raw,
                "refined": refined,
                "mask": mask,
                "residual": residual_visualization,
                "uncertainty": uncertainty,
            }
            if redistortion is not None:
                products = {
                    name: _redistort_tensor(value, redistortion)
                    for name, value in products.items()
                }
            name = _output_name(camera.image_name)
            map_name = f"{Path(name).stem}.png"
            save_tensor_image(products["raw"], directories["raw"] / name)
            save_tensor_image(products["refined"], directories["refined"] / name)
            save_tensor_image(
                products["residual"], directories["residual"] / map_name
            )
            save_tensor_image(
                products["uncertainty"], directories["uncertainty"] / map_name
            )
            if save_mask:
                save_tensor_image(products["mask"], directories["mask"] / map_name)
    print(f"Saved raw and refined renders to {output_root}")


if __name__ == "__main__":
    main()
