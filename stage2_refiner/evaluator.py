"""Evaluation, region diagnostics and comparison image export."""

import csv
import math
import os
import torch

from .losses import ssim_metric
from .utils import save_rgb


def image_metrics(pred, target, lpips_model=None):
    mse = torch.mean((pred - target) ** 2)
    metrics = {"psnr": -10.0 * torch.log10(mse.clamp_min(1e-10)), "ssim": ssim_metric(pred, target),
               "l1": torch.mean((pred - target).abs()), "mse": mse}
    metrics["lpips"] = lpips_model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean() if lpips_model is not None else None
    return metrics


def _average(rows, prefix):
    result = {}
    for key in ("psnr", "ssim", "lpips", "l1", "mse"):
        values = [row[f"{prefix}_{key}"] for row in rows if row.get(f"{prefix}_{key}") is not None]
        result[f"{prefix}_{key}"] = sum(values) / len(values) if values else None
    return result


def summarize_by_scene(rows):
    scenes = {}
    for row in rows: scenes.setdefault(row["scene"], []).append(row)
    output = {}
    for scene, scene_rows in scenes.items():
        summary = {**_average(scene_rows, "stage1"), **_average(scene_rows, "stage2")}
        summary["delta_psnr"] = summary["stage2_psnr"] - summary["stage1_psnr"]
        summary["delta_ssim"] = summary["stage2_ssim"] - summary["stage1_ssim"]
        if summary["stage1_lpips"] is not None:
            summary["delta_lpips"] = summary["stage2_lpips"] - summary["stage1_lpips"]
        output[scene] = summary
    return output


def _masked_psnr(pred, target, mask):
    if not mask.any(): return None
    error = (pred - target).square().mean(1, keepdim=True)
    mse = error[mask].mean()
    return (-10.0 * torch.log10(mse.clamp_min(1e-10))).item()


def region_diagnostics(rgb, refined, target, depth, normal, alpha):
    baseline_error = (rgb - target).abs().mean(1, keepdim=True)
    median = baseline_error.flatten(1).median(1).values[:, None, None, None]
    depth_gx = (depth[..., :, 1:] - depth[..., :, :-1]).abs(); depth_gy = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    normal_gx = (normal[..., :, 1:] - normal[..., :, :-1]).abs().mean(1, keepdim=True)
    normal_gy = (normal[..., 1:, :] - normal[..., :-1, :]).abs().mean(1, keepdim=True)
    edge = torch.zeros_like(alpha); edge[..., :, 1:] += depth_gx + normal_gx; edge[..., 1:, :] += depth_gy + normal_gy
    threshold = torch.quantile(edge.flatten(1).float(), .75, dim=1).to(edge.dtype)[:, None, None, None]
    masks = {"low_residual": baseline_error <= median, "high_residual": baseline_error > median,
             "geometry_edge": edge > threshold, "non_edge": edge <= threshold,
             "low_alpha": alpha < .5, "high_alpha": alpha >= .5}
    result = {}
    for name, mask in masks.items():
        before, after = _masked_psnr(rgb, target, mask), _masked_psnr(refined, target, mask)
        result[f"region_{name}_stage1_psnr"] = before; result[f"region_{name}_stage2_psnr"] = after
        result[f"region_{name}_delta_psnr"] = after - before if before is not None and after is not None else None
    return result


@torch.no_grad()
def evaluate_loader(model, loader, device, output_dir=None, tile_fn=None, lpips_model=None):
    model.eval(); rows = []
    folders = ("input_stage1", "output_stage2", "ground_truth", "error_stage1",
               "error_stage2", "residual_prediction", "comparison")
    if output_dir:
        for folder in folders: os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    for batch in loader:
        tensors = {key: batch[key].to(device) for key in ("rgb", "depth", "normal", "alpha", "target")}
        if tile_fn: refined, residual = tile_fn(model, tensors["rgb"], tensors["depth"], tensors["normal"], tensors["alpha"])
        else: refined, residual = model(rgb=tensors["rgb"], depth=tensors["depth"], normal=tensors["normal"], alpha=tensors["alpha"], return_residual=True)
        for index in range(refined.shape[0]):
            before = image_metrics(tensors["rgb"][index:index+1], tensors["target"][index:index+1], lpips_model)
            after = image_metrics(refined[index:index+1], tensors["target"][index:index+1], lpips_model)
            row = {"scene": batch["scene"][index], "image_name": batch["image_name"][index]}
            row.update({f"stage1_{key}": value.item() if value is not None else None for key, value in before.items()})
            row.update({f"stage2_{key}": value.item() if value is not None else None for key, value in after.items()})
            row["delta_psnr"] = row["stage2_psnr"] - row["stage1_psnr"]
            row["delta_ssim"] = row["stage2_ssim"] - row["stage1_ssim"]
            row["delta_lpips"] = row["stage2_lpips"] - row["stage1_lpips"] if row["stage1_lpips"] is not None else None
            row.update(region_diagnostics(tensors["rgb"][index:index+1], refined[index:index+1],
                                          tensors["target"][index:index+1], tensors["depth"][index:index+1],
                                          tensors["normal"][index:index+1], tensors["alpha"][index:index+1]))
            rows.append(row)
            if output_dir:
                name = f"{len(rows)-1:05d}.png"
                target = tensors["target"][index]; rgb = tensors["rgb"][index]; result = refined[index]
                save_rgb(rgb, os.path.join(output_dir, "input_stage1", name)); save_rgb(result, os.path.join(output_dir, "output_stage2", name)); save_rgb(target, os.path.join(output_dir, "ground_truth", name))
                save_rgb((rgb-target).abs(), os.path.join(output_dir, "error_stage1", name)); save_rgb((result-target).abs(), os.path.join(output_dir, "error_stage2", name))
                save_rgb((residual[index] / (2 * model.max_residual) + .5), os.path.join(output_dir, "residual_prediction", name))
                save_rgb(torch.cat((rgb, result, target), dim=2), os.path.join(output_dir, "comparison", name))
    if not rows: return {}, []
    summary = {**_average(rows, "stage1"), **_average(rows, "stage2")}
    summary["delta_psnr"] = summary["stage2_psnr"] - summary["stage1_psnr"]
    summary["delta_ssim"] = summary["stage2_ssim"] - summary["stage1_ssim"]
    summary["delta_lpips"] = summary["stage2_lpips"] - summary["stage1_lpips"] if summary["stage1_lpips"] is not None else None
    if output_dir:
        with open(os.path.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return summary, rows
