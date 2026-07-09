import argparse
import csv
import os
from argparse import Namespace
from pathlib import Path

import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.eval_utils import (
    average_metric_dicts,
    calculate_render_metrics,
    get_lpips_model,
    metrics_to_floats,
)
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


def evaluate_scene(scene_path, model_path, iteration, resolution, psnr_max):
    dataset = make_dataset_args(scene_path, model_path, resolution)
    pipeline = make_pipeline_args()

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        lpips_model = get_lpips_model()

        metric_values = []
        for viewpoint in tqdm(scene.getTestCameras(), desc=f"Evaluating {scene_path.name}"):
            if not getattr(viewpoint, "has_ground_truth", True):
                continue
            image = torch.clamp(
                render(
                    viewpoint,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )["render"],
                0.0,
                1.0,
            )
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            metric_values.append(calculate_render_metrics(image, gt_image, psnr_max, lpips_model))

    return metrics_to_floats(average_metric_dicts(metric_values))


def main():
    parser = argparse.ArgumentParser(description="Evaluate public Phase1 scenes with PSNR/SSIM/LPIPS/Score.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--psnr_max", type=float, default=30.0)
    parser.add_argument("--csv_path", default="metrics_public.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    scenes = discover_scenes(args.data_root)
    if not scenes:
        raise SystemExit(f"No supported scenes found under {args.data_root}")

    rows = []
    for scene_path in scenes:
        model_path = Path(args.model_root) / scene_path.name
        if not model_path.exists():
            print(f"Skip {scene_path.name}: model folder not found at {model_path}")
            continue
        metrics = evaluate_scene(scene_path, model_path, args.iteration, args.resolution, args.psnr_max)
        if metrics is None:
            print(f"{scene_path.name}: Ground-truth test images not found, skip metrics for private set.")
            continue
        row = {"scene": scene_path.name, **metrics}
        rows.append(row)

    fieldnames = ["scene", "psnr", "ssim", "lpips", "psnr_norm", "score"]
    with open(args.csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print("\nscene, psnr, ssim, lpips, psnr_norm, score")
    for row in rows:
        print(
            f"{row['scene']}, {row['psnr']:.6f}, {row['ssim']:.6f}, "
            f"{row['lpips'] if row['lpips'] is not None else 'skipped'}, "
            f"{row['psnr_norm']:.6f}, {row['score'] if row['score'] is not None else 'skipped'}"
        )

    scored_rows = [row for row in rows if row.get("score") is not None]
    if scored_rows:
        mean_score = sum(row["score"] for row in scored_rows) / len(scored_rows)
        print(f"\nMean score: {mean_score:.6f}")
    print(f"Saved metrics to {args.csv_path}")


if __name__ == "__main__":
    main()
