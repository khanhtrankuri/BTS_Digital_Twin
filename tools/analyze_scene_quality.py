"""Per-camera BTS scene diagnostics without modifying training data or poses."""
import csv
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.eval_utils import calculate_render_metrics, get_lpips_model, metrics_to_floats
from utils.general_utils import safe_state


def laplacian_variance(image):
    gray = 0.2989 * image[:1] + 0.5870 * image[1:2] + 0.1140 * image[2:3]
    kernel = image.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    return F.conv2d(gray[None], kernel, padding=1).var().item()


def main(dataset, pipeline, iteration, output_dir, psnr_max):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        lpips_model = get_lpips_model()
        rows = []
        for index, camera in enumerate(scene.getTrainCameras()):
            package = render(camera, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp,
                             render_geometry=True)
            image, gt = package["render"].clamp(0, 1), camera.original_image[:3].cuda()
            metrics = metrics_to_floats(calculate_render_metrics(image, gt, psnr_max, lpips_model))
            error = (image - gt).abs()
            stem = f"{index:04d}_{camera.image_name}"
            torchvision.utils.save_image(gt, os.path.join(output_dir, stem + "_gt.png"))
            torchvision.utils.save_image(image, os.path.join(output_dir, stem + "_render.png"))
            torchvision.utils.save_image(error, os.path.join(output_dir, stem + "_abs_error.png"))
            torchvision.utils.save_image(package["alpha"], os.path.join(output_dir, stem + "_alpha.png"))
            rows.append({"camera_id": camera.uid, "image_name": camera.image_name, **metrics,
                         "mse": float((image - gt).square().mean()), "brightness": float(gt.mean()),
                         "rgb_std": float(gt.std()), "blur_score": laplacian_variance(gt),
                         "edge_density": float((error.mean(0) > 0.1).float().mean()),
                         "alpha_coverage": float((package["alpha"] > 1e-3).float().mean()),
                         "camera_x": float(camera.camera_center[0]), "camera_y": float(camera.camera_center[1]),
                         "camera_z": float(camera.camera_center[2])})
    with open(os.path.join(output_dir, "camera_quality.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render and report quality per training camera.")
    model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1); parser.add_argument("--output_dir", required=True)
    parser.add_argument("--psnr_max", type=float, default=30.0); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser); safe_state(args.quiet)
    main(model.extract(args), pipeline.extract(args), args.iteration, args.output_dir, args.psnr_max)
