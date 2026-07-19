"""Edge, thin-structure, sky, and hallucination diagnostics for v4."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from utils.geometry_losses import sobel_edge_map


def _mask(mask, reference):
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask[None]
    return mask.to(reference.device) > 0.5


def _binary_chamfer(predicted: torch.Tensor, target: torch.Tensor) -> float:
    pred = predicted.detach().cpu().numpy().astype(np.uint8)[0]
    gt = target.detach().cpu().numpy().astype(np.uint8)[0]
    if not pred.any() or not gt.any():
        return float("inf") if pred.any() != gt.any() else 0.0
    distance_to_gt = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 3)
    distance_to_pred = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 3)
    return float(0.5 * (distance_to_gt[pred > 0].mean() + distance_to_pred[gt > 0].mean()))


def edge_metrics(predicted_rgb: torch.Tensor, target_rgb: torch.Tensor,
                 region_mask=None, threshold: float = 0.20, eps: float = 1e-6) -> dict[str, float]:
    predicted = sobel_edge_map(predicted_rgb)
    target = sobel_edge_map(target_rgb)
    valid = _mask(region_mask, target)
    pred_binary = (predicted >= threshold) & valid
    target_binary = (target >= threshold) & valid
    true_positive = (pred_binary & target_binary).sum().float()
    precision = true_positive / pred_binary.sum().clamp_min(1)
    recall = true_positive / target_binary.sum().clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    p, t = predicted[valid], target[valid]
    correlation = (((p - p.mean()) * (t - t.mean())).mean()
                   / (p.std(unbiased=False) * t.std(unbiased=False) + eps)) if p.numel() else p.new_tensor(0.0)
    return {"gradient_mae": float((p - t).abs().mean().item()) if p.numel() else 0.0,
            "sobel_correlation": float(correlation.item()), "edge_precision": float(precision.item()),
            "edge_recall": float(recall.item()), "edge_f1": float(f1.item()),
            "edge_chamfer": _binary_chamfer(pred_binary, target_binary)}


def _skeleton(binary: np.ndarray) -> np.ndarray:
    image = binary.astype(np.uint8).copy()
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while image.any():
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton |= image & (1 - opened)
        image = cv2.erode(image, element)
    return skeleton


def thin_structure_metrics(predicted_rgb: torch.Tensor, target_rgb: torch.Tensor,
                           region_mask=None, threshold: float = 0.20) -> dict[str, float]:
    pred_edge = (sobel_edge_map(predicted_rgb) >= threshold).detach().cpu().numpy()[0]
    target_edge = (sobel_edge_map(target_rgb) >= threshold).detach().cpu().numpy()[0]
    if region_mask is not None:
        valid = _mask(region_mask, sobel_edge_map(target_rgb)).cpu().numpy()[0]
        pred_edge &= valid
        target_edge &= valid
    pred_skeleton = torch.from_numpy(_skeleton(pred_edge))[None].bool()
    target_skeleton = torch.from_numpy(_skeleton(target_edge))[None].bool()
    intersection = (pred_skeleton & target_skeleton).sum().float()
    precision = intersection / pred_skeleton.sum().clamp_min(1)
    recall = intersection / target_skeleton.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    pred_components = cv2.connectedComponents(pred_skeleton[0].numpy().astype(np.uint8))[0] - 1
    target_components = cv2.connectedComponents(target_skeleton[0].numpy().astype(np.uint8))[0] - 1
    return {"skeleton_precision": float(precision), "skeleton_recall": float(recall),
            "skeleton_f1": float(f1), "skeleton_chamfer": _binary_chamfer(pred_skeleton, target_skeleton),
            "connectivity_error": float(abs(pred_components - target_components))}


def sky_metrics(predicted_rgb: torch.Tensor, target_rgb: torch.Tensor, sky_mask: torch.Tensor,
                alpha: torch.Tensor | None = None, depth: torch.Tensor | None = None,
                eps: float = 1e-8) -> dict[str, float]:
    mask = _mask(sky_mask, predicted_rgb[:1]).expand_as(predicted_rgb)
    mse = (predicted_rgb[mask] - target_rgb[mask]).square().mean() if mask.any() else predicted_rgb.new_tensor(0.0)
    output = {"sky_psnr": float((-10.0 * torch.log10(mse + eps)).item()),
              "sky_gradient_energy": float(sobel_edge_map(predicted_rgb)[mask[:1]].mean().item()) if mask.any() else 0.0}
    sky_1 = mask[:1]
    if alpha is not None:
        alpha_values = alpha[sky_1]
        output["sky_alpha_mass"] = float(alpha_values.mean().item()) if alpha_values.numel() else 0.0
        output["sky_floater_score"] = float((alpha_values > 0.05).float().mean().item()) if alpha_values.numel() else 0.0
    if depth is not None:
        values = depth[sky_1]
        output["sky_depth_variance"] = float(values.var(unbiased=False).item()) if values.numel() else 0.0
    return output


def unsupported_new_edge_rate(stage1_rgb: torch.Tensor, refined_rgb: torch.Tensor,
                              supported_edge_mask: torch.Tensor, threshold: float = 0.20) -> float:
    before = sobel_edge_map(stage1_rgb) >= threshold
    after = sobel_edge_map(refined_rgb) >= threshold
    supported = _mask(supported_edge_mask, before)
    new_edges = after & ~before
    return float((new_edges & ~supported).sum().float() / new_edges.sum().clamp_min(1))


def image_region_masks(height: int, width: int, device=None) -> dict[str, torch.Tensor]:
    center = torch.zeros((1, height, width), dtype=torch.bool, device=device)
    y0, y1, x0, x1 = height // 4, height - height // 4, width // 4, width - width // 4
    center[:, y0:y1, x0:x1] = True
    return {"center": center, "outer_25_percent": ~center}
