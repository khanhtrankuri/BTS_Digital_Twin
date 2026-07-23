"""Stage-1 multi-view RGB reprojection and exposure-robust losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.depth_reprojection import pairwise_depth_consistency


def _zero_with_grad(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def _cfg(cfg, name: str, default):
    return getattr(cfg, name, default)


def warp_source_rgb_to_target(
    target_depth: torch.Tensor,
    target_camera,
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    source_camera,
    target_alpha: torch.Tensor | None,
    source_alpha: torch.Tensor | None,
    sky_mask: torch.Tensor | None,
    cfg,
) -> dict[str, torch.Tensor]:
    """Warp source RGB using target camera-z depth and depth visibility checks."""

    consistency = pairwise_depth_consistency(
        target_depth, source_depth, target_camera, source_camera,
        target_alpha=target_alpha, source_alpha=source_alpha, sky_mask=sky_mask,
        sigma_z=float(_cfg(cfg, "multiview_sigma_z", 0.02)),
        relative_threshold=float(_cfg(cfg, "multiview_relative_depth_threshold", 0.05)),
        min_alpha=float(_cfg(cfg, "multiview_min_alpha", 0.01)))
    if source_rgb.ndim != 3 or source_rgb.shape[0] != 3:
        raise ValueError("source_rgb must have shape [3,H,W]")
    grid = consistency.sampling_grid
    warped = F.grid_sample(
        source_rgb.to(device=target_depth.device, dtype=target_depth.dtype)[None],
        grid[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    return {
        "warped_rgb": warped,
        "valid_mask": consistency.hard_visibility,
        "depth_confidence": consistency.soft_confidence,
        "relative_depth_error": consistency.relative_error,
        "sampling_grid": grid,
    }


def _local_standardize(image: torch.Tensor, patch_size: int, eps: float) -> torch.Tensor:
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    padding = patch_size // 2
    batched = image[None]
    padded = F.pad(batched, (padding, padding, padding, padding), mode="reflect")
    mean = F.avg_pool2d(padded, patch_size, stride=1)
    second = F.avg_pool2d(padded.square(), patch_size, stride=1)
    std = (second - mean.square()).clamp_min(0.0).add(eps).sqrt()
    return ((batched - mean) / std)[0]


def _gradient(image: torch.Tensor) -> torch.Tensor:
    gray = 0.2989 * image[:1] + 0.5870 * image[1:2] + 0.1140 * image[2:3]
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray[None], kernel_x, padding=1)[0]
    gy = F.conv2d(gray[None], kernel_x.transpose(-1, -2), padding=1)[0]
    return torch.cat((gx, gy), dim=0)


def _local_zncc(first: torch.Tensor, second: torch.Tensor, patch_size: int,
                eps: float) -> torch.Tensor:
    weights = first.new_tensor((0.2989, 0.5870, 0.1140))[:, None, None]
    first_gray = (first * weights).sum(dim=0, keepdim=True)[None]
    second_gray = (second * weights).sum(dim=0, keepdim=True)[None]
    padding = patch_size // 2
    first_padded = F.pad(first_gray, (padding,) * 4, mode="reflect")
    second_padded = F.pad(second_gray, (padding,) * 4, mode="reflect")
    mean_first = F.avg_pool2d(first_padded, patch_size, stride=1)
    mean_second = F.avg_pool2d(second_padded, patch_size, stride=1)
    second_first = F.avg_pool2d(first_padded.square(), patch_size, stride=1)
    second_second = F.avg_pool2d(second_padded.square(), patch_size, stride=1)
    cross = F.avg_pool2d(first_padded * second_padded, patch_size, stride=1)
    variance_first = (second_first - mean_first.square()).clamp_min(0.0)
    variance_second = (second_second - mean_second.square()).clamp_min(0.0)
    covariance = cross - mean_first * mean_second
    # Clamp before sqrt. sqrt(0).clamp_min(eps) is finite forward but its
    # backward can evaluate inf * 0 on flat patches and poison exposure/SH.
    denominator = torch.sqrt((variance_first * variance_second).clamp_min(eps ** 2))
    correlation = covariance / denominator
    both_flat = (variance_first < eps) & (variance_second < eps)
    same_flat = (mean_first - mean_second).abs() < eps
    return torch.where(both_flat, same_flat.to(first.dtype), correlation).clamp(-1.0, 1.0)[0]


def multiview_rgb_loss(
    target_rgb: torch.Tensor,
    warped_source_rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    cfg,
) -> dict[str, torch.Tensor]:
    """Compute local-ZNCC, gradient and normalized Charbonnier components."""

    if target_rgb.shape != warped_source_rgb.shape or target_rgb.ndim != 3:
        raise ValueError("target and warped RGB must have matching [3,H,W] shapes")
    mask = valid_mask.bool() & torch.isfinite(confidence) & (confidence > 0)
    mask &= torch.isfinite(target_rgb).all(dim=0, keepdim=True)
    mask &= torch.isfinite(warped_source_rgb).all(dim=0, keepdim=True)
    minimum = int(_cfg(cfg, "multiview_min_valid_pixels", 256))
    zero = _zero_with_grad(target_rgb)
    if int(mask.sum().item()) < minimum:
        return {"total": zero, "zncc": zero, "gradient": zero, "charbonnier": zero,
                "valid_fraction": mask.float().mean()}

    eps = float(_cfg(cfg, "multiview_charbonnier_eps", 1e-3))
    patch_size = int(_cfg(cfg, "multiview_patch_size", 7))
    target_safe = torch.nan_to_num(target_rgb, nan=0.0, posinf=0.0, neginf=0.0)
    source_safe = torch.nan_to_num(warped_source_rgb, nan=0.0, posinf=0.0, neginf=0.0)
    target_normalized = _local_standardize(target_safe, patch_size, eps)
    source_normalized = _local_standardize(source_safe, patch_size, eps)
    weights = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0) * mask
    denominator = weights.sum().clamp_min(1e-6)

    correlation = torch.nan_to_num(
        _local_zncc(target_safe, source_safe, patch_size, 1e-6),
        nan=0.0, posinf=0.0, neginf=0.0)
    zncc = (weights * (1.0 - correlation)).sum() / denominator
    target_gradient = _gradient(target_normalized)
    source_gradient = _gradient(source_normalized)
    gradient_error = (target_gradient - source_gradient).abs().mean(dim=0, keepdim=True)
    gradient = (weights * gradient_error).sum() / denominator
    charbonnier_error = torch.sqrt(
        (target_normalized - source_normalized).square().mean(dim=0, keepdim=True) + eps ** 2) - eps
    charbonnier = (weights * charbonnier_error).sum() / denominator
    total = (
        float(_cfg(cfg, "multiview_zncc_weight", 1.0)) * zncc
        + float(_cfg(cfg, "multiview_gradient_weight", 0.20)) * gradient
        + float(_cfg(cfg, "multiview_charbonnier_weight", 0.10)) * charbonnier
    )
    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    return {"total": total, "zncc": torch.nan_to_num(zncc),
            "gradient": torch.nan_to_num(gradient),
            "charbonnier": torch.nan_to_num(charbonnier),
            "valid_fraction": mask.float().mean()}
