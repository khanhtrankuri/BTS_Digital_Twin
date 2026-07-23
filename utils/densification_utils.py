"""Numerically robust score helpers for BTS-GeoGS-v3 densification."""

from __future__ import annotations

import torch


def split_opacity_conserving(
    raw_opacity: torch.Tensor,
    num_outputs: int,
) -> torch.Tensor:
    """Return a child logit that conserves parent transmittance.

    If ``num_outputs`` identical children replace one parent with alpha
    ``alpha_p``, each child receives
    ``1 - (1 - alpha_p) ** (1 / num_outputs)``. The computation uses
    ``log1p``/``expm1`` and dtype-aware clamps so saturated logits never
    produce NaN or Inf.
    """

    if num_outputs < 1:
        raise ValueError("num_outputs must be positive")
    if not torch.is_floating_point(raw_opacity):
        raise TypeError("raw_opacity must be a floating-point tensor")
    alpha_parent = torch.sigmoid(raw_opacity)
    finfo = torch.finfo(raw_opacity.dtype)
    alpha_parent = alpha_parent.clamp(min=finfo.eps, max=1.0 - finfo.eps)
    log_transmittance = torch.log1p(-alpha_parent) / float(num_outputs)
    alpha_child = -torch.expm1(log_transmittance)
    alpha_child = alpha_child.clamp(min=finfo.eps, max=1.0 - finfo.eps)
    return torch.log(alpha_child) - torch.log1p(-alpha_child)


def conservative_opacity_prune_mask(
    opacity: torch.Tensor,
    visibility_count: torch.Tensor,
    gaussian_age: torch.Tensor,
    min_opacity: float,
    min_age: int,
) -> torch.Tensor:
    """Prune only mature, unsupported low-opacity Gaussians.

    Opacity reset and opacity-conserving splits intentionally create low alpha.
    Treating alpha alone as a deletion signal can therefore collapse the whole
    scene one window after reset.
    """

    opacity = opacity.reshape(-1)
    visibility_count = visibility_count.reshape(-1)
    gaussian_age = gaussian_age.reshape(-1)
    if not (opacity.shape == visibility_count.shape == gaussian_age.shape):
        raise ValueError("opacity, visibility_count and gaussian_age must align")
    return ((opacity < float(min_opacity))
            & (visibility_count <= 0)
            & (gaussian_age >= int(min_age)))


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
