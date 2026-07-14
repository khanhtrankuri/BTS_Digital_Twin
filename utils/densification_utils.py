"""Numerically robust score helpers for BTS-GeoGS-v3 densification."""

from __future__ import annotations

import torch


def robust_normalize_score(values: torch.Tensor, valid_mask: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = valid_mask.bool() & torch.isfinite(values)
    result = torch.zeros_like(values)
    if not valid_mask.any():
        return result
    valid = values[valid_mask]
    median = torch.median(valid)
    q1 = torch.quantile(valid, 0.25)
    q3 = torch.quantile(valid, 0.75)
    iqr = q3 - q1
    if not torch.isfinite(iqr) or iqr <= eps:
        return result
    result[valid_mask] = ((valid - median) / (iqr + eps)).clamp(0.0, 10.0)
    return result


def corrected_residual_map(corrected_render: torch.Tensor, gt_image: torch.Tensor,
                           residual_type: str = "charbonnier", eps: float = 1e-6) -> torch.Tensor:
    diff = corrected_render.detach() - gt_image.detach()
    if residual_type == "l1":
        residual = diff.abs().mean(dim=0, keepdim=True)
    elif residual_type == "charbonnier":
        residual = torch.sqrt(diff.square().mean(dim=0, keepdim=True) + eps)
    else:
        raise ValueError(f"Unsupported residual type: {residual_type}")
    return torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)


def percentile_mask(scores: torch.Tensor, valid_mask: torch.Tensor,
                    percentile: float) -> torch.Tensor:
    valid_mask = valid_mask.bool() & torch.isfinite(scores)
    result = torch.zeros_like(valid_mask)
    if not valid_mask.any():
        return result
    percentile = float(max(0.0, min(1.0, percentile)))
    threshold = torch.quantile(scores[valid_mask], percentile)
    result[valid_mask] = scores[valid_mask] >= threshold
    return result


def limit_mask(mask: torch.Tensor, score: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0:
        return torch.zeros_like(mask)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if indices.numel() <= limit:
        return mask
    selected = indices[torch.topk(score[indices], k=int(limit)).indices]
    result = torch.zeros_like(mask)
    result[selected] = True
    return result


def accumulate_visible_statistics(accumulator: torch.Tensor, denominator: torch.Tensor,
                                  values: torch.Tensor, visible_mask: torch.Tensor) -> None:
    visible_mask = visible_mask.reshape(-1).bool()
    if visible_mask.numel() != accumulator.shape[0]:
        raise ValueError("visible_mask must contain one entry per Gaussian")
    if visible_mask.any():
        accumulator[visible_mask] += values[visible_mask]
        denominator[visible_mask] += 1
