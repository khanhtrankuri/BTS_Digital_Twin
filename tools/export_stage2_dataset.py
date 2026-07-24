"""Export frozen Gaussian renders and geometry for BTS GeoNAF-GS Phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# Allow ``python tools/export_stage2_dataset.py`` from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.stage2_gaussian import (
    discover_scene_directories,
    load_gaussian_scene,
    resolve_model_directory,
    serialize_camera,
)
from utils.stage2_io import load_stage2_config, save_tensor_image


GEOMETRY_CHANNELS = {
    "depth": 1,
    "normal": 3,
    "alpha": 1,
    "uncertainty": 1,
    "depth_variance": 1,
}


def validate_geometry_arrays(
    arrays: dict[str, np.ndarray], height: int, width: int
) -> None:
    for name, channels in GEOMETRY_CHANNELS.items():
        if name not in arrays:
            raise KeyError(f"Renderer did not provide geometry map {name!r}")
        value = np.asarray(arrays[name])
        if value.shape != (channels, height, width):
            raise ValueError(
                f"{name} must have shape {(channels, height, width)}, "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if (arrays["depth_variance"] < 0).any():
        raise ValueError("depth_variance contains negative values")
    if arrays["alpha"].min() < -1e-5 or arrays["alpha"].max() > 1.0 + 1e-5:
        raise ValueError("alpha lies outside [0,1]")
    if (
        arrays["uncertainty"].min() < -1e-5
        or arrays["uncertainty"].max() > 1.0 + 1e-5
    ):
        raise ValueError("uncertainty lies outside [0,1]")
    if arrays["normal"].min() < -1.0 - 1e-4 or arrays["normal"].max() > 1.0 + 1e-4:
        raise ValueError("normal lies outside the expected [-1,1] range")


def stage2_files_complete(
    rgb_path: Path, gt_path: Path, geometry_path: Path
) -> bool:
    if not (rgb_path.is_file() and gt_path.is_file() and geometry_path.is_file()):
        return False
    try:
        with Image.open(rgb_path) as rgb_image, Image.open(gt_path) as gt_image:
            rgb_image.verify()
            gt_image.verify()
        with Image.open(rgb_path) as rgb_image, Image.open(gt_path) as gt_image:
            if rgb_image.size != gt_image.size:
                return False
            width, height = rgb_image.size
        with np.load(geometry_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in GEOMETRY_CHANNELS}
            validate_geometry_arrays(arrays, height, width)
        return True
    except (OSError, ValueError, KeyError):
        return False


def deterministic_validation_indices(
    count: int, scene: str, validation_fraction: float, seed: int
) -> set[int]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0,1)")
    validation_count = int(round(count * validation_fraction))
    if validation_fraction > 0.0 and count > 1:
        validation_count = max(1, min(count - 1, validation_count))
    digest = int(hashlib.sha256(scene.encode("utf-8")).hexdigest()[:8], 16)
    indices = list(range(count))
    random.Random(int(seed) + digest).shuffle(indices)
    return set(indices[:validation_count])


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _save_geometry(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    converted = {
        name: value.astype(np.float16) for name, value in arrays.items()
    }
    sample = converted["depth"]
    validate_geometry_arrays(converted, sample.shape[-2], sample.shape[-1])
    np.savez_compressed(
        temporary,
        **converted,
    )
    os.replace(temporary, path)


def export_scene(
    scene_path: Path,
    model_path: Path,
    output_root: Path,
    iteration: int,
    config: dict[str, Any],
    *,
    device: str,
    resolution: int,
    overwrite: bool,
    exposure_compensation: bool,
) -> Path:
    from gaussian_renderer import render

    dataset, pipeline, _, gaussians, scene, background = load_gaussian_scene(
        scene_path,
        model_path,
        iteration,
        resolution=resolution,
        device=device,
    )
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError(f"Scene {scene_path.name} has no training cameras")
    scene_output = output_root / scene_path.name
    for directory in ("rgb", "gt", "geometry", "cameras"):
        (scene_output / directory).mkdir(parents=True, exist_ok=True)

    data_config = config["DATA"]
    validation_fraction = 1.0 - float(data_config.get("TRAIN_SPLIT", 0.9))
    validation = deterministic_validation_indices(
        len(cameras),
        scene_path.name,
        validation_fraction,
        int(data_config.get("SEED", 42)),
    )
    frames: list[dict[str, Any]] = []
    for index, camera in enumerate(
        tqdm(cameras, desc=f"Exporting {scene_path.name}")
    ):
        stem = f"{index:05d}_{Path(camera.image_name).stem}"
        rgb_path = scene_output / "rgb" / f"{stem}.png"
        gt_path = scene_output / "gt" / f"{stem}.png"
        geometry_path = scene_output / "geometry" / f"{stem}.npz"
        camera_path = scene_output / "cameras" / f"{stem}.json"
        complete = stage2_files_complete(rgb_path, gt_path, geometry_path)
        if overwrite or not complete:
            with torch.no_grad():
                package = render(
                    camera,
                    gaussians,
                    pipeline,
                    background,
                    render_geometry=True,
                    apply_exposure=exposure_compensation,
                    exposure_mode="training",
                )
            gaussian_rgb = package["render"].detach().float().clamp(0.0, 1.0)
            ground_truth = camera.original_image[:3].detach().float().cpu()
            if dataset.train_test_exp:
                gaussian_rgb = gaussian_rgb[..., gaussian_rgb.shape[-1] // 2 :]
                ground_truth = ground_truth[..., ground_truth.shape[-1] // 2 :]
            if gaussian_rgb.shape != ground_truth.shape:
                raise ValueError(
                    f"Render/GT mismatch for {camera.image_name}: "
                    f"{tuple(gaussian_rgb.shape)} vs {tuple(ground_truth.shape)}"
                )
            height, width = gaussian_rgb.shape[-2:]
            arrays = {
                name: package[name].detach().float().cpu().numpy()
                for name in GEOMETRY_CHANNELS
            }
            validate_geometry_arrays(arrays, height, width)
            save_tensor_image(gaussian_rgb, rgb_path)
            save_tensor_image(ground_truth, gt_path)
            _save_geometry(geometry_path, arrays)
        else:
            with Image.open(rgb_path) as image:
                width, height = image.size

        camera_metadata = serialize_camera(camera, index)
        camera_path.write_text(
            json.dumps(camera_metadata, indent=2), encoding="utf-8"
        )
        frames.append(
            {
                "scene": scene_path.name,
                "image_name": camera.image_name,
                "rgb_path": _relative(rgb_path, scene_output),
                "gt_path": _relative(gt_path, scene_output),
                "geometry_path": _relative(geometry_path, scene_output),
                "camera_path": _relative(camera_path, scene_output),
                "width": int(width),
                "height": int(height),
                "split": "val" if index in validation else "train",
                **camera_metadata,
            }
        )

    manifest = {
        "format": "bts-geonaf-stage2-v1",
        "scene": scene_path.name,
        "gaussian_model": str(model_path),
        "gaussian_iteration": int(scene.loaded_iter),
        "normalization": config["NORMALIZATION"],
        "frames": frames,
    }
    manifest_path = scene_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frozen AbsGS renders for GeoNAF Stage 2"
    )
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument(
        "--config", default="configs/stage2/geonaf_base.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exposure_compensation", action="store_true")
    args = parser.parse_args()

    config = load_stage2_config(args.config)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for scene_path in discover_scene_directories(args.scene_root):
        model_path = resolve_model_directory(args.model_root, scene_path.name)
        if args.iteration >= 0:
            checkpoint = (
                model_path
                / "point_cloud"
                / f"iteration_{args.iteration}"
                / "point_cloud.ply"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Gaussian checkpoint does not exist: {checkpoint}"
                )
        manifests.append(
            export_scene(
                scene_path,
                model_path,
                output_root,
                args.iteration,
                config,
                device=args.device,
                resolution=args.resolution,
                overwrite=args.overwrite,
                exposure_compensation=args.exposure_compensation,
            )
        )
    print("Exported manifests:")
    for path in manifests:
        print(f"  {path}")


if __name__ == "__main__":
    main()
