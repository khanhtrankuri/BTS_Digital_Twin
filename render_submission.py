import argparse
import os
import zipfile
from argparse import Namespace
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

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


def make_dataset_args(source_path, model_path, resolution):
    return Namespace(
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
    )


def make_pipeline_args():
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )


def render_scene(scene_path, model_path, iteration, resolution, output_dir,
                 exposure_compensation=False, test_exposure_mode="identity"):
    dataset = make_dataset_args(scene_path, model_path, resolution)
    pipeline = make_pipeline_args()
    scene_output = Path(output_dir) / scene_path.name
    scene_output.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
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
    parser.add_argument("--test_exposure_mode", choices=["identity", "nearest_camera", "weighted_nearest"], default="identity")
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
                     args.exposure_compensation, args.test_exposure_mode)

    zip_submission(args.output_dir, args.zip_path)
    print(f"Saved submission zip to {args.zip_path}")


if __name__ == "__main__":
    main()
