"""Compare raw Gaussian and refined Stage-2 output per image and per scene."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.stage2_dataset import Stage2Dataset
from utils.stage2_inference import tiled_refine
from utils.stage2_io import (
    load_refiner_checkpoint,
    load_stage2_config,
    save_tensor_image,
)
from utils.stage2_losses import load_lpips_if_available, stage2_metrics


def _average(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float | None]:
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BTS GeoNAF-GS Stage 2")
    parser.add_argument("--manifest_root", required=True)
    parser.add_argument("--refiner_checkpoint", required=True)
    parser.add_argument(
        "--config", default="configs/stage2/geonaf_base.yaml"
    )
    parser.add_argument("--output_dir", default="evaluation_stage2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--allow_weight_download", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true")
    args = parser.parse_args()

    config = load_stage2_config(args.config)
    device = torch.device(args.device)
    dataset = Stage2Dataset(
        args.manifest_root,
        config,
        split=None if args.split == "all" else args.split,
        training=False,
    )
    model, _ = load_refiner_checkpoint(
        args.refiner_checkpoint, config, device
    )
    model.eval()
    lpips_model = (
        None
        if args.disable_lpips
        else load_lpips_if_available(
            device, allow_weight_download=args.allow_weight_download
        )
    )
    if not args.disable_lpips and lpips_model is None:
        print(
            "LPIPS skipped: package/backbone weights are not available locally. "
            "Use --allow_weight_download to permit a download."
        )
    output_root = Path(args.output_dir)
    comparison_root = output_root / "comparisons"
    output_root.mkdir(parents=True, exist_ok=True)
    inference_config = config["INFERENCE"]
    tile_size = int(inference_config.get("TILE_SIZE", 512))
    overlap = int(inference_config.get("TILE_OVERLAP", 64))
    amp_enabled = device.type == "cuda"
    rows: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        stage2_input = sample["input"].unsqueeze(0).to(device)
        gaussian_rgb = sample["gaussian_rgb"].unsqueeze(0).to(device)
        target = sample["gt"].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with torch.no_grad(), context:
            output = tiled_refine(
                model, stage2_input, tile_size=tile_size, overlap=overlap
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_vram = 0.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_metrics = stage2_metrics(
            gaussian_rgb.float(), target.float(), lpips_model=lpips_model
        )
        refined_metrics = stage2_metrics(
            output["final_rgb"].float(),
            target.float(),
            lpips_model=lpips_model,
        )
        correction = output["correction"].float()
        scale = float(config["MODEL"]["RESIDUAL_SCALE"])
        magnitude = correction.abs().amax(dim=1, keepdim=True)
        over_limit_fraction = float((magnitude >= 0.9 * scale).float().mean().item())
        row: dict[str, Any] = {
            "scene": sample["scene"],
            "image_name": sample["image_name"],
            "inference_time_ms": elapsed_ms,
            "peak_vram_mb": peak_vram,
            "mean_abs_correction": float(correction.abs().mean().item()),
            "p95_abs_correction": float(
                torch.quantile(correction.abs().flatten(), 0.95).item()
            ),
            "near_residual_limit_fraction": over_limit_fraction,
        }
        for name, value in raw_metrics.items():
            row[f"raw_{name}"] = value
        for name, value in refined_metrics.items():
            row[f"refined_{name}"] = value
            raw_value = raw_metrics[name]
            row[f"delta_{name}"] = (
                None
                if value is None or raw_value is None
                else float(value - raw_value)
            )
        rows.append(row)

        scene_dir_name = str(sample["scene"])
        stem = Path(str(sample["image_name"])).stem
        save_tensor_image(
            target[0], comparison_root / "gt" / scene_dir_name / f"{stem}.png"
        )
        save_tensor_image(
            gaussian_rgb[0],
            comparison_root / "raw" / scene_dir_name / f"{stem}.png",
        )
        save_tensor_image(
            output["final_rgb"][0],
            comparison_root / "refined" / scene_dir_name / f"{stem}.png",
        )
        residual_vis = (0.5 + output["correction"][0] / (2.0 * scale)).clamp(
            0.0, 1.0
        )
        save_tensor_image(
            residual_vis,
            comparison_root / "residual" / scene_dir_name / f"{stem}.png",
        )
        save_tensor_image(
            output["effective_mask"][0],
            comparison_root / "mask" / scene_dir_name / f"{stem}.png",
        )

    fieldnames = list(rows[0])
    with (output_root / "metrics_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = [
        key
        for key in fieldnames
        if key not in {"scene", "image_name"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene"])].append(row)
    per_scene = {
        scene: {
            "num_images": len(scene_rows),
            **_average(scene_rows, metric_keys),
        }
        for scene, scene_rows in grouped.items()
    }
    (output_root / "metrics_per_scene.json").write_text(
        json.dumps(per_scene, indent=2), encoding="utf-8"
    )
    averaged = _average(rows, metric_keys)
    regression_scenes = {
        "psnr": [
            scene
            for scene, metrics in per_scene.items()
            if metrics.get("delta_psnr") is not None
            and metrics["delta_psnr"] < 0.0
        ],
        "ssim": [
            scene
            for scene, metrics in per_scene.items()
            if metrics.get("delta_ssim") is not None
            and metrics["delta_ssim"] < 0.0
        ],
        "lpips": [
            scene
            for scene, metrics in per_scene.items()
            if metrics.get("delta_lpips") is not None
            and metrics["delta_lpips"] > 0.0
        ],
    }
    aggregate = {
        "num_images": len(rows),
        **averaged,
        "psnr_improved": bool((averaged.get("delta_psnr") or 0.0) > 0.0),
        "ssim_improved": bool((averaged.get("delta_ssim") or 0.0) > 0.0),
        "lpips_improved": (
            None
            if averaged.get("delta_lpips") is None
            else bool(averaged["delta_lpips"] < 0.0)
        ),
        "regression_scenes": regression_scenes,
        "overcorrection_diagnostic": {
            "definition": (
                "fraction of pixels where max-channel absolute correction "
                "reaches at least 90% of RESIDUAL_SCALE"
            ),
            "mean_fraction": averaged.get("near_residual_limit_fraction"),
        },
    }
    (output_root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
