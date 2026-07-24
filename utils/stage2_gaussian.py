"""Read-only adapters between the AbsGS stage and standalone Stage 2 tools."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from arguments import load_model_config_defaults, load_model_optimization_args


def is_scene_directory(path: Path) -> bool:
    return (
        (path / "train" / "sparse").exists()
        and (path / "test" / "test_poses.csv").exists()
    ) or (path / "sparse").exists() or (path / "transforms_train.json").exists()


def discover_scene_directories(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {root}")
    if is_scene_directory(root):
        return [root.resolve()]
    scenes = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and is_scene_directory(path)
    )
    if not scenes:
        raise FileNotFoundError(f"No supported scene found under {root}")
    return scenes


def resolve_model_directory(model_root: str | Path, scene_name: str) -> Path:
    root = Path(model_root)
    candidate = root / scene_name
    if (candidate / "point_cloud").is_dir():
        return candidate.resolve()
    if (root / "point_cloud").is_dir():
        return root.resolve()
    raise FileNotFoundError(
        f"No Gaussian point_cloud directory for {scene_name} under {root}"
    )


def make_stage1_dataset_args(
    source_path: str | Path,
    model_path: str | Path,
    *,
    resolution: int = 1,
    device: str = "cuda",
    config_defaults: dict[str, Any] | None = None,
) -> Namespace:
    defaults = config_defaults or {}
    dataset = Namespace(
        sh_degree=3,
        source_path=str(Path(source_path).resolve()),
        model_path=str(Path(model_path).resolve()),
        images="images",
        depths="",
        resolution=int(resolution),
        white_background=False,
        train_test_exp=False,
        data_device=device,
        eval=True,
        depth_prior_dir="",
        normal_prior_dir="",
        confidence_prior_dir="",
        camera_use_undistorted_data=False,
        camera_strict_model_validation=True,
        camera_warn_if_distortion_dropped=True,
        camera_require_undistortion_metadata=False,
        validation_split_file="",
        strict_sparse_path="",
    )
    for name in (
        "depth_prior_dir",
        "normal_prior_dir",
        "confidence_prior_dir",
        "camera_use_undistorted_data",
        "camera_strict_model_validation",
        "camera_warn_if_distortion_dropped",
        "camera_require_undistortion_metadata",
        "validation_split_file",
        "strict_sparse_path",
        "white_background",
        "train_test_exp",
    ):
        if name in defaults:
            setattr(dataset, name, defaults[name])
    return dataset


def make_stage1_pipeline_args(
    config_defaults: dict[str, Any] | None = None,
) -> Namespace:
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=bool((config_defaults or {}).get("antialiasing", False)),
    )


def load_gaussian_scene(
    source_path: str | Path,
    model_path: str | Path,
    iteration: int,
    *,
    resolution: int = 1,
    device: str = "cuda",
):
    """Load a trained scene without changing its checkpoint or Stage-1 config."""

    if not str(device).startswith("cuda"):
        raise ValueError(
            "The bundled Gaussian rasterizer is CUDA-only; use --device cuda"
        )
    from scene import GaussianModel, Scene

    defaults = load_model_config_defaults(model_path)
    dataset = make_stage1_dataset_args(
        source_path,
        model_path,
        resolution=resolution,
        device=device,
        config_defaults=defaults,
    )
    pipeline = make_stage1_pipeline_args(defaults)
    optimization = load_model_optimization_args(model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=int(iteration),
        shuffle=False,
        optimization_args=optimization,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    return dataset, pipeline, optimization, gaussians, scene, background


def serialize_camera(camera, index: int) -> dict[str, Any]:
    """Serialize intrinsics and conventional column-vector W2C extrinsics."""

    extrinsics = camera.world_view_transform.detach().float().cpu().t()
    camera_to_world = torch.linalg.inv(extrinsics)
    view_direction = camera_to_world[:3, 2]
    view_direction = view_direction / view_direction.norm().clamp_min(1e-8)
    return {
        "camera_index": int(index),
        "intrinsics": {
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
        },
        "extrinsics": extrinsics.tolist(),
        "camera_center": camera.camera_center.detach().float().cpu().tolist(),
        "view_direction": view_direction.tolist(),
        "camera_convention": "column-vector world_to_camera; +z forward; RGB CHW",
    }
