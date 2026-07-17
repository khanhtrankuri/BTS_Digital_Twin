"""Render Stage 1 target views and optionally apply the Stage 2 refiner."""

from argparse import ArgumentParser
import os
from pathlib import Path

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from stage2_refiner.checkpoint import load_checkpoint
from stage2_refiner.geometry import preprocess_geometry, tiled_inference
from stage2_refiner.utils import load_config, model_from_config, save_rgb


def main():
    parser = ArgumentParser(description=__doc__); model_args = ModelParams(parser, sentinel=True); pipeline_args = PipelineParams(parser)
    parser.add_argument("--stage1_iteration", type=int, default=-1); parser.add_argument("--refiner_checkpoint")
    parser.add_argument("--refiner_config"); parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test"); parser.add_argument("--disable_refiner", action="store_true")
    parser.add_argument("--exposure_compensation", action="store_true"); parser.add_argument("--test_exposure_mode", choices=["identity", "nearest_camera", "weighted_nearest"], default="identity")
    args = get_combined_args(parser)
    if not args.disable_refiner and not args.refiner_checkpoint: raise ValueError("--refiner_checkpoint is required unless --disable_refiner is set")
    if not torch.cuda.is_available(): raise RuntimeError("Stage 1 rendering requires CUDA")
    dataset, pipe = model_args.extract(args), pipeline_args.extract(args); dataset.eval = args.split == "test"
    gaussians = GaussianModel(dataset.sh_degree); scene = Scene(dataset, gaussians, load_iteration=args.stage1_iteration, shuffle=False)
    cameras = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    background = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device="cuda")
    refiner = cfg = None
    if not args.disable_refiner:
        raw = torch.load(args.refiner_checkpoint, map_location="cpu", weights_only=False)
        cfg = load_config(args.refiner_config) if args.refiner_config else raw.get("config")
        if cfg is None: raise ValueError("Checkpoint has no config; pass --refiner_config")
        refiner = model_from_config(cfg).cuda().eval(); refiner.load_state_dict(raw.get("ema", raw["model"]))
    output_dir = Path(args.output_dir); (output_dir / "stage1").mkdir(parents=True, exist_ok=True); (output_dir / "refined").mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for camera in tqdm(cameras, desc=f"Rendering {args.split}"):
            package = render(camera, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp,
                             render_geometry=not args.disable_refiner, apply_exposure=args.exposure_compensation,
                             exposure_mode="training" if args.split == "train" else args.test_exposure_mode)
            rgb = package["render"].clamp(0, 1); save_rgb(rgb, str(output_dir / "stage1" / camera.image_name))
            if args.disable_refiner: refined = rgb
            else:
                geo_cfg, infer_cfg = cfg.get("GEOMETRY", {}), cfg.get("INFERENCE", {})
                depth, normal, alpha, _ = preprocess_geometry(package["depth"], package["normal"], package["alpha"],
                    geo_cfg.get("DEPTH_NORMALIZATION", "robust_per_view"), geo_cfg.get("ALPHA_THRESHOLD", 0.01))
                refined, _ = tiled_inference(refiner, rgb.unsqueeze(0), depth.unsqueeze(0), normal.unsqueeze(0), alpha.unsqueeze(0),
                                              int(infer_cfg.get("TILE_SIZE", 512)), int(infer_cfg.get("TILE_OVERLAP", 32)))
                refined = refined[0]
            save_rgb(refined, str(output_dir / "refined" / camera.image_name))
    print(f"Saved {len(cameras)} views to {output_dir}")


if __name__ == "__main__": main()
