"""Optional Stage-2 camera pairing and occlusion-aware forward reprojection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F


def select_nearest_camera_pairs(
    entries: list[dict[str, Any]],
    *,
    distance_weight: float = 1.0,
    direction_weight: float = 1.0,
) -> dict[int, int]:
    """Select one nearby, similarly oriented camera within each scene."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        grouped[str(entry.get("scene", ""))].append(index)
    pairs: dict[int, int] = {}
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        centers = torch.tensor(
            [entries[index]["camera_center"] for index in indices], dtype=torch.float32
        )
        directions = F.normalize(
            torch.tensor(
                [entries[index]["view_direction"] for index in indices],
                dtype=torch.float32,
            ),
            dim=1,
        )
        distances = torch.cdist(centers, centers)
        positive = distances[distances > 0]
        scale = positive.median() if positive.numel() else distances.new_tensor(1.0)
        angle_cost = 1.0 - directions @ directions.t()
        scores = (
            float(distance_weight) * distances / scale.clamp_min(1e-6)
            + float(direction_weight) * angle_cost
        )
        scores.fill_diagonal_(float("inf"))
        for local_index, global_index in enumerate(indices):
            pairs[global_index] = indices[int(scores[local_index].argmin().item())]
    return pairs


def _sample_map(
    value: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    height, width = value.shape[-2:]
    grid = torch.stack(
        (
            2.0 * (u + 0.5) / float(width) - 1.0,
            2.0 * (v + 0.5) / float(height) - 1.0,
        ),
        dim=-1,
    )
    grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    return F.grid_sample(
        value[None],
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, :, 0]


def forward_warp_rgb(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    *,
    source_alpha: torch.Tensor | None = None,
    target_alpha: torch.Tensor | None = None,
    source_uncertainty: torch.Tensor | None = None,
    target_uncertainty: torch.Tensor | None = None,
    dynamic_mask: torch.Tensor | None = None,
    target_dynamic_mask: torch.Tensor | None = None,
    min_alpha: float = 0.01,
    max_uncertainty: float = 0.8,
    relative_depth_threshold: float = 0.05,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-splat source RGB into the target with rendered-depth occlusion.

    Extrinsics are conventional column-vector world-to-camera matrices.
    Gradients flow to source RGB while camera/depth geometry acts as guidance.
    """

    if source_rgb.ndim != 3 or source_rgb.shape[0] != 3:
        raise ValueError("source_rgb must have shape [3,H,W]")
    for name, depth in (("source_depth", source_depth), ("target_depth", target_depth)):
        if depth.ndim != 3 or depth.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,H,W]")
    device, dtype = source_rgb.device, source_rgb.dtype
    source_depth = torch.nan_to_num(source_depth.to(device, dtype), nan=0.0)
    target_depth = torch.nan_to_num(target_depth.to(device, dtype), nan=0.0)
    source_height, source_width = source_depth.shape[-2:]
    target_height, target_width = target_depth.shape[-2:]
    fx_s, fy_s, cx_s, cy_s = (
        source_intrinsics.to(device=device, dtype=dtype).unbind()
    )
    fx_t, fy_t, cx_t, cy_t = (
        target_intrinsics.to(device=device, dtype=dtype).unbind()
    )
    v, u = torch.meshgrid(
        torch.arange(source_height, device=device, dtype=dtype),
        torch.arange(source_width, device=device, dtype=dtype),
        indexing="ij",
    )
    z = source_depth[0]
    x = (u + 0.5 - cx_s) / fx_s * z
    y = (v + 0.5 - cy_s) / fy_s * z
    camera_points = torch.stack((x, y, z, torch.ones_like(z)), dim=0).reshape(4, -1)
    source_c2w = torch.linalg.inv(source_extrinsics.to(device=device, dtype=dtype))
    world = source_c2w @ camera_points
    target_camera = target_extrinsics.to(device=device, dtype=dtype) @ world
    target_z = target_camera[2]
    target_u = fx_t * target_camera[0] / target_z.clamp_min(eps) + cx_t - 0.5
    target_v = fy_t * target_camera[1] / target_z.clamp_min(eps) + cy_t - 0.5

    valid = torch.isfinite(target_u) & torch.isfinite(target_v)
    valid &= torch.isfinite(target_z) & (z.reshape(-1) > eps) & (target_z > eps)
    valid &= (
        (target_u >= 0)
        & (target_u <= target_width - 1)
        & (target_v >= 0)
        & (target_v <= target_height - 1)
    )
    sampled_depth = _sample_map(target_depth, target_u, target_v)[0]
    depth_error = (sampled_depth - target_z).abs() / sampled_depth.abs().clamp_min(eps)
    valid &= (sampled_depth > eps) & (depth_error <= float(relative_depth_threshold))

    if source_alpha is not None:
        valid &= source_alpha.to(device, dtype).reshape(-1) >= float(min_alpha)
    if target_alpha is not None:
        valid &= (
            _sample_map(target_alpha.to(device, dtype), target_u, target_v)[0]
            >= float(min_alpha)
        )
    if source_uncertainty is not None:
        valid &= (
            source_uncertainty.to(device, dtype).reshape(-1)
            <= float(max_uncertainty)
        )
    if target_uncertainty is not None:
        valid &= (
            _sample_map(
                target_uncertainty.to(device, dtype), target_u, target_v
            )[0]
            <= float(max_uncertainty)
        )
    if dynamic_mask is not None:
        valid &= dynamic_mask.to(device).reshape(-1) < 0.5
    if target_dynamic_mask is not None:
        valid &= (
            _sample_map(target_dynamic_mask.to(device, dtype), target_u, target_v)[0]
            < 0.5
        )

    left = torch.floor(target_u).long()
    top = torch.floor(target_v).long()
    du = target_u - left.to(dtype)
    dv = target_v - top.to(dtype)
    candidates: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    colors: list[torch.Tensor] = []
    source_flat = source_rgb.reshape(3, -1)
    for offset_x, offset_y, weight in (
        (0, 0, (1.0 - du) * (1.0 - dv)),
        (1, 0, du * (1.0 - dv)),
        (0, 1, (1.0 - du) * dv),
        (1, 1, du * dv),
    ):
        current_x = left + offset_x
        current_y = top + offset_y
        current_valid = valid & (current_x >= 0) & (current_x < target_width)
        current_valid &= (current_y >= 0) & (current_y < target_height)
        current_weight = weight[current_valid]
        candidates.append(current_y[current_valid] * target_width + current_x[current_valid])
        weights.append(current_weight)
        colors.append(source_flat[:, current_valid] * current_weight[None])

    if not candidates or sum(item.numel() for item in candidates) == 0:
        return (
            source_rgb.new_zeros((3, target_height, target_width)),
            source_rgb.new_zeros((1, target_height, target_width)),
        )
    indices = torch.cat(candidates)
    splat_weights = torch.cat(weights)
    splat_colors = torch.cat(colors, dim=1)
    flat_pixels = target_height * target_width
    accumulated_weight = source_rgb.new_zeros(flat_pixels).scatter_add(
        0, indices, splat_weights
    )
    accumulated_rgb = source_rgb.new_zeros((3, flat_pixels)).scatter_add(
        1, indices[None].expand(3, -1), splat_colors
    )
    warped = accumulated_rgb / accumulated_weight.clamp_min(eps)[None]
    mask = accumulated_weight > eps
    return warped.reshape(3, target_height, target_width), mask.reshape(
        1, target_height, target_width
    ).to(dtype)


def masked_multiview_l1(
    warped_source: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.bool()
    valid &= torch.isfinite(warped_source).all(dim=0, keepdim=True)
    valid &= torch.isfinite(target_rgb).all(dim=0, keepdim=True)
    if not valid.any():
        return target_rgb.sum() * 0.0
    error = (warped_source - target_rgb).abs().mean(dim=0, keepdim=True)
    return error[valid].mean()
