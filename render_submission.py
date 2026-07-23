import argparse
import json
import os
import zipfile
from argparse import Namespace
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm
import cv2

from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import load_model_config_defaults, load_model_optimization_args
from utils.general_utils import safe_state
from tools.prepare_undistorted_scene import redistort_render

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def is_scene_dir(path):
    return (
        (path / "train" / "sparse").exists() and (path / "test" / "test_poses.csv").exists()
    ) or (path / "sparse").exists() or (path / "transforms_train.json").exists()


def discover_scenes(data_root):
    root = Path(data_root)
    if not root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")
    if is_scene_dir(root):
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir() and is_scene_dir(path)])


def make_dataset_args(source_path, model_path, resolution, config_defaults=None):
    dataset = Namespace(
        sh_degree=3,
        source_path=str(source_path.resolve()),
        model_path=str(model_path),
        images="images",
        depths="",
        resolution=resolution,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
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
    if config_defaults:
        for name in (
            "depth_prior_dir", "normal_prior_dir", "confidence_prior_dir",
            "camera_use_undistorted_data", "camera_strict_model_validation",
            "camera_warn_if_distortion_dropped", "camera_require_undistortion_metadata",
            "validation_split_file", "strict_sparse_path",
        ):
            if name in config_defaults:
                setattr(dataset, name, config_defaults[name])
    return dataset


def make_pipeline_args(config_defaults=None):
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=bool((config_defaults or {}).get("antialiasing", False)),
    )


def render_scene(scene_path, model_path, iteration, resolution, output_dir,
                 exposure_compensation=False, test_exposure_mode="identity",
                 redistort_to_source_grid=False):
    config_defaults = load_model_config_defaults(model_path)
    dataset = make_dataset_args(scene_path, model_path, resolution, config_defaults)
    pipeline = make_pipeline_args(config_defaults)
    optimization_args = load_model_optimization_args(model_path)
    if test_exposure_mode == "auto":
        test_exposure_mode = (config_defaults or {}).get(
            "test_exposure_mode", "identity")
    scene_output = Path(output_dir) / scene_path.name
    scene_output.mkdir(parents=True, exist_ok=True)
    redistortion_metadata = None
    if redistort_to_source_grid:
        metadata_path = Path(scene_path) / "undistortion_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Redistortion requested but metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cameras = metadata.get("cameras", {})
        if len(cameras) != 1:
            raise ValueError(
                "Submission redistortion currently requires one shared camera model per scene")
        redistortion_metadata = next(iter(cameras.values()))

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset, gaussians, load_iteration=iteration, shuffle=False,
            optimization_args=optimization_args)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for viewpoint in tqdm(scene.getTestCameras(), desc=f"Rendering {scene_path.name}"):
            rendering = torch.clamp(
                render(
                    viewpoint,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                    apply_exposure=exposure_compensation,
                    exposure_mode=test_exposure_mode,
                )["render"],
                0.0,
                1.0,
            )
            image_path = scene_output / viewpoint.image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(rendering, str(image_path))
            if redistortion_metadata is not None:
                encoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if encoded is None:
                    raise IOError(f"OpenCV could not read rendered image: {image_path}")
                encoded = redistort_render(encoded, redistortion_metadata)
                if not cv2.imwrite(str(image_path), encoded):
                    raise IOError(f"OpenCV could not write redistorted image: {image_path}")


def zip_submission(output_dir, zip_path):
    output_dir = Path(output_dir)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(output_dir).as_posix())


def main():
    parser = argparse.ArgumentParser(description="Render private Phase1 test poses and zip submission images.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--output_dir", default="submission")
    parser.add_argument("--zip_path", default="submission.zip")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exposure_compensation", action="store_true")
    parser.add_argument("--test_exposure_mode", choices=["identity", "nearest_camera", "weighted_nearest",
                        "pose_confidence_blend", "temporal_weighted", "pose_temporal_weighted",
                        "temporal_spline", "auto"], default="identity")
    parser.add_argument("--skip_zip", action="store_true",
                        help="Render images but leave ZIP creation to an outer workflow.")
    parser.add_argument("--redistort_to_source_grid", action="store_true",
                        help="Map prepared undistorted renders back to the original test pixel grid.")
    args = parser.parse_args()

    safe_state(args.quiet)
    scenes = discover_scenes(args.data_root)
    if not scenes:
        raise SystemExit(f"No supported scenes found under {args.data_root}")

    for scene_path in scenes:
        model_path = Path(args.model_root) / scene_path.name
        if not model_path.exists():
            print(f"Skip {scene_path.name}: model folder not found at {model_path}")
            continue
        render_scene(scene_path, model_path, args.iteration, args.resolution, args.output_dir,
                     args.exposure_compensation, args.test_exposure_mode,
                     args.redistort_to_source_grid)

    if not args.skip_zip:
        zip_submission(args.output_dir, args.zip_path)
        print(f"Saved submission zip to {args.zip_path}")


if __name__ == "__main__":
    main()
