"""Export frozen Stage 1 RGB/depth/normal/alpha renders for Stage 2 training."""

from argparse import ArgumentParser, Namespace
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from stage2_refiner.utils import save_rgb, write_json


def deterministic_holdout(cameras, ratio, seed):
    ranked = sorted(cameras, key=lambda camera: hashlib.sha256(f"{seed}:{camera.image_name}".encode()).hexdigest())
    val_count = min(max(1, round(len(ranked) * ratio)), max(1, len(ranked) - 1)) if len(ranked) > 1 else 0
    val_names = {camera.image_name for camera in ranked[:val_count]}
    return ([camera for camera in cameras if camera.image_name not in val_names],
            [camera for camera in cameras if camera.image_name in val_names])


def is_scene_dir(path):
    path = Path(path)
    return ((path / "train" / "sparse").exists() and (path / "test" / "test_poses.csv").exists()
            or (path / "sparse").exists()
            or (path / "transforms_train.json").exists())


def discover_scene_dirs(source_path):
    root = Path(source_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source path does not exist: {root}")
    if is_scene_dir(root):
        return [root], False
    scenes = sorted(child.resolve() for child in root.iterdir() if child.is_dir() and is_scene_dir(child))
    if not scenes:
        raise FileNotFoundError(f"No supported scenes found under: {root}")
    return scenes, True


def _model_defaults():
    parser = ArgumentParser(add_help=False)
    ModelParams(parser, sentinel=False)
    return vars(parser.parse_args([]))


def scene_arguments(cli_args, source_path, model_path):
    """Merge each Stage 1 cfg_args with explicit exporter CLI overrides."""
    merged = _model_defaults()
    cfg_path = Path(model_path) / "cfg_args"
    if cfg_path.exists():
        text = cfg_path.read_text(encoding="utf-8")
        saved = eval(text, {"Namespace": Namespace, "__builtins__": {}})
        merged.update(vars(saved))
    for key, value in vars(cli_args).items():
        if value is not None:
            merged[key] = value
    merged["source_path"] = str(Path(source_path).resolve())
    merged["model_path"] = str(Path(model_path).resolve())
    return Namespace(**merged)


def tensor_matrix(value):
    return value.detach().cpu().numpy().tolist() if torch.is_tensor(value) else np.asarray(value).tolist()


def export_camera(camera, split, camera_id, scene_name, output_dir, gaussians, pipe, background,
                  train_test_exp=False, apply_exposure=False):
    package = render(camera, gaussians, pipe, background, use_trained_exp=train_test_exp,
                     render_geometry=True, apply_exposure=apply_exposure,
                     exposure_mode="training" if split == "train" else "identity")
    rgb, gt = package["render"].clamp(0, 1), camera.original_image[:3].clamp(0, 1)
    if train_test_exp:
        rgb, gt = rgb[..., rgb.shape[-1] // 2:], gt[..., gt.shape[-1] // 2:]
    stem = f"{camera_id:06d}"
    paths = {name: output_dir / split / name for name in ("rgb_render", "depth", "normal", "alpha", "rgb_gt", "metadata")}
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    rgb_path, gt_path = paths["rgb_render"] / f"{stem}.png", paths["rgb_gt"] / f"{stem}.png"
    depth_path, normal_path, alpha_path = paths["depth"] / f"{stem}.npy", paths["normal"] / f"{stem}.npy", paths["alpha"] / f"{stem}.npy"
    metadata_path = paths["metadata"] / f"{stem}.json"
    save_rgb(rgb, str(rgb_path)); save_rgb(gt, str(gt_path))
    np.save(depth_path, package["depth"].detach().cpu().numpy().astype(np.float32))
    np.save(normal_path, package["normal"].detach().cpu().numpy().astype(np.float32))
    np.save(alpha_path, package["alpha"].detach().cpu().numpy().astype(np.float32))
    fx = camera.image_width / (2.0 * np.tan(camera.FoVx / 2.0)); fy = camera.image_height / (2.0 * np.tan(camera.FoVy / 2.0))
    metadata = {"scene": scene_name, "split": split, "camera_id": camera_id, "source_uid": int(camera.uid),
                "image_name": camera.image_name, "width": int(rgb.shape[-1]), "height": int(rgb.shape[-2]),
                "intrinsics": [[float(fx), 0.0, float(camera.cx if camera.cx is not None else camera.image_width/2)],
                               [0.0, float(fy), float(camera.cy if camera.cy is not None else camera.image_height/2)], [0.0, 0.0, 1.0]],
                "world_view_transform": tensor_matrix(camera.world_view_transform)}
    write_json(metadata, str(metadata_path))
    relative = lambda path: os.path.relpath(path, output_dir).replace("\\", "/")
    return {"scene": scene_name, "split": split, "camera_id": camera_id, "image_name": camera.image_name,
            "rgb_render": relative(rgb_path), "depth": relative(depth_path), "normal": relative(normal_path),
            "alpha": relative(alpha_path), "rgb_gt": relative(gt_path), "metadata": relative(metadata_path),
            "width": int(rgb.shape[-1]), "height": int(rgb.shape[-2])}


def export_scene(cli_args, scene_path, model_path, output_dir, requested, pipeline_params):
    scene_args = scene_arguments(cli_args, scene_path, model_path)
    dataset, pipe = ModelParams.extract(pipeline_params[0], scene_args), PipelineParams.extract(pipeline_params[1], scene_args)
    dataset.eval = bool(cli_args.use_official_val)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=cli_args.iteration, shuffle=False)
    train_cameras = [camera for camera in scene.getTrainCameras() if camera.has_ground_truth]
    official_val = [camera for camera in scene.getTestCameras() if camera.has_ground_truth] if cli_args.use_official_val else []
    if official_val:
        val_cameras = official_val
    else:
        train_cameras, val_cameras = deterministic_holdout(train_cameras, cli_args.val_ratio, cli_args.split_seed)
    if not train_cameras or not val_cameras:
        raise RuntimeError(f"{Path(scene_path).name}: need at least one GT camera in both Stage 2 train and val splits")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    scene_name = Path(dataset.source_path).resolve().name
    samples, camera_id = [], 0
    with torch.no_grad():
        for split, cameras in (("train", train_cameras), ("val", val_cameras)):
            if split not in requested:
                continue
            for camera in tqdm(cameras, desc=f"{scene_name}: {split}"):
                samples.append(export_camera(
                    camera, split, camera_id, scene_name, output_dir, gaussians, pipe, background,
                    dataset.train_test_exp, cli_args.exposure_compensation))
                camera_id += 1
    payload = {
        "version": 1,
        "stage1_model_path": os.path.abspath(dataset.model_path),
        "stage1_iteration": scene.loaded_iter,
        "source_path": os.path.abspath(dataset.source_path),
        "split_seed": cli_args.split_seed,
        "val_ratio": cli_args.val_ratio,
        "leakage_policy": "test cameras excluded; train and val camera ids are disjoint",
        "samples": samples,
    }
    write_json(payload, str(output_dir / "manifest.json"))
    del scene, gaussians, background
    torch.cuda.empty_cache()
    return payload


def rebase_scene_samples(samples, scene_directory):
    path_fields = ("rgb_render", "depth", "normal", "alpha", "rgb_gt", "metadata")
    rebased = []
    for sample in samples:
        item = dict(sample)
        for field in path_fields:
            item[field] = f"{scene_directory}/{item[field]}".replace("\\", "/")
        rebased.append(item)
    return rebased


def main():
    parser = ArgumentParser(
        description=__doc__,
        epilog=("For all scenes, pass -s <dataset_parent>, -m <stage1_output_root>, and "
                "--output_dir <stage2_data_root>. Each scene is matched by directory name."))
    model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1); parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train,val", help="Comma-separated subset of train,val")
    parser.add_argument("--val_ratio", type=float, default=0.10); parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--use_official_val", action="store_true", help="Use Stage 1 test cameras as val only when GT exists")
    parser.add_argument("--exposure_compensation", action="store_true")
    args = parser.parse_args(); requested = {item.strip() for item in args.split.split(",") if item.strip()}
    if not requested <= {"train", "val"}: raise ValueError("--split supports only train,val; test GT is never exported")
    if not 0 < args.val_ratio < 1: raise ValueError("--val_ratio must be between 0 and 1")
    if not torch.cuda.is_available(): raise RuntimeError("Stage 1 dataset export requires CUDA and the 3DGS rasterizer")
    scenes, parent_mode = discover_scene_dirs(args.source_path)
    model_root, output_root = Path(args.model_path).resolve(), Path(args.output_dir).resolve()
    jobs = []
    for scene_path in scenes:
        scene_model = model_root / scene_path.name if parent_mode else model_root
        scene_output = output_root / scene_path.name if parent_mode else output_root
        if not scene_model.is_dir():
            raise FileNotFoundError(
                f"Stage 1 model is missing for scene '{scene_path.name}': {scene_model}\n"
                "Expected the train_multi_cuda.py layout: <model_root>/<scene_name>.")
        jobs.append((scene_path, scene_model, scene_output))

    print(f"Discovered {len(jobs)} scene(s). Exporting sequentially on the active CUDA device.")
    combined_samples, scene_records = [], []
    for index, (scene_path, scene_model, scene_output) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {scene_path.name}: {scene_model} -> {scene_output}")
        payload = export_scene(args, scene_path, scene_model, scene_output, requested, (model, pipeline))
        if parent_mode:
            combined_samples.extend(rebase_scene_samples(payload["samples"], scene_path.name))
        else:
            combined_samples.extend(payload["samples"])
        scene_records.append({"scene": scene_path.name, "source_path": str(scene_path),
                              "stage1_model_path": str(scene_model),
                              "stage1_iteration": payload["stage1_iteration"],
                              "samples": len(payload["samples"])})

    if parent_mode:
        combined = {"version": 1, "multi_scene": True, "scenes": scene_records,
                    "split_seed": args.split_seed, "val_ratio": args.val_ratio,
                    "leakage_policy": "test cameras excluded; train and val are disjoint within every scene",
                    "samples": combined_samples}
        write_json(combined, str(output_root / "manifest.json"))
        print(f"Exported {len(combined_samples)} samples from {len(scene_records)} scenes to {output_root / 'manifest.json'}")
    else:
        print(f"Exported {len(combined_samples)} samples to {output_root / 'manifest.json'}")


if __name__ == "__main__": main()
