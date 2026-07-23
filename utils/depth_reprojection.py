"""Shared metric z-depth reprojection for Stage 1 and future Stage 2.

Transforms use the repository's row-vector convention:
``world_h @ camera.world_view_transform -> camera_h``. Depth is camera-space
z-depth, not Euclidean ray length. Pixel sampling uses ``align_corners=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DepthConsistencyResult:
    projected_depth: torch.Tensor
    sampled_source_depth: torch.Tensor
    relative_error: torch.Tensor
    hard_visibility: torch.Tensor
    soft_confidence: torch.Tensor
    sampling_grid: torch.Tensor


def _depth_2d(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3 and depth.shape[0] == 1:
        return depth[0]
    if depth.ndim == 2:
        return depth
    raise ValueError("depth must have shape [H,W] or [1,H,W]")


def unproject_z_depth(depth: torch.Tensor, camera) -> torch.Tensor:
    """Unproject target z-depth to world points with shape ``[H,W,3]``."""

    depth = _depth_2d(depth)
    height, width = depth.shape
    device, dtype = depth.device, depth.dtype
    v, u = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return unproject_pixels(u, v, depth, camera)


def unproject_pixels(u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor, camera) -> torch.Tensor:
    """Unproject arbitrary pixel-center indices and camera z-depth to world XYZ."""

    if u.shape != v.shape or u.shape != depth.shape:
        raise ValueError("u, v and depth must have identical shapes")
    device, dtype = depth.device, depth.dtype
    x = (u + 0.5 - float(camera.cx)) / float(camera.fx) * depth
    y = (v + 0.5 - float(camera.cy)) / float(camera.fy) * depth
    camera_points = torch.stack((x, y, depth, torch.ones_like(depth)), dim=-1)
    camera_to_world = torch.linalg.inv(camera.world_view_transform.to(device=device, dtype=dtype))
    world = camera_points @ camera_to_world
    return world[..., :3] / world[..., 3:].clamp_min(1e-8)


def project_world_points(points: torch.Tensor, camera) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ``[...,3]`` world points to pixel u/v and camera z-depth."""

    if points.shape[-1] != 3:
        raise ValueError("points must end in three XYZ coordinates")
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat((points, ones), dim=-1)
    camera_points = homogeneous @ camera.world_view_transform.to(device=points.device, dtype=points.dtype)
    z = camera_points[..., 2]
    u = float(camera.fx) * camera_points[..., 0] / z.clamp_min(1e-8) + float(camera.cx) - 0.5
    v = float(camera.fy) * camera_points[..., 1] / z.clamp_min(1e-8) + float(camera.cy) - 0.5
    return u, v, z


def pixel_grid(u: torch.Tensor, v: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert pixel-center coordinates to a grid_sample grid."""

    x = 2.0 * (u + 0.5) / float(width) - 1.0
    y = 2.0 * (v + 0.5) / float(height) - 1.0
    return torch.stack((x, y), dim=-1)


def pairwise_depth_consistency(target_depth: torch.Tensor, source_depth: torch.Tensor,
                               target_camera, source_camera, *, target_alpha=None,
                               source_alpha=None, sky_mask=None, sigma_z: float = 0.02,
                               relative_threshold: float = 0.05,
                               min_alpha: float = 0.01, eps: float = 1e-6) -> DepthConsistencyResult:
    """Reproject target depth into a source rendered-depth map."""

    target_raw = _depth_2d(target_depth)
    finite_target = torch.isfinite(target_raw)
    target = torch.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)
    source = _depth_2d(source_depth).to(device=target.device, dtype=target.dtype)
    world = unproject_z_depth(target, target_camera)
    u, v, projected_z = project_world_points(world, source_camera)
    raw_grid = pixel_grid(u, v, source.shape[1], source.shape[0])
    # Invalid/behind-camera points can produce inf coordinates. grid_sample may
    # return a finite zero forward but an undefined grid gradient backward, so
    # sanitize before sampling rather than masking only after projection.
    grid = torch.nan_to_num(raw_grid, nan=2.0, posinf=2.0, neginf=-2.0)
    source_safe = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    sampled = F.grid_sample(source_safe[None, None], grid[None], mode="bilinear",
                            padding_mode="zeros", align_corners=False)[0, 0]
    projected_safe = torch.nan_to_num(projected_z, nan=0.0, posinf=0.0, neginf=0.0)
    relative = torch.nan_to_num(
        torch.abs(projected_safe - sampled) / sampled.abs().clamp_min(eps),
        nan=1e6, posinf=1e6, neginf=1e6)
    inside = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    finite = (finite_target & torch.isfinite(projected_z)
              & torch.isfinite(raw_grid).all(dim=-1))
    valid = finite & (target > eps) & (projected_z > eps) & (sampled > eps) & inside
    if target_alpha is not None:
        valid &= _depth_2d(target_alpha) >= min_alpha
    if source_alpha is not None:
        source_alpha_2d = _depth_2d(source_alpha).to(device=target.device, dtype=target.dtype)
        sampled_alpha = F.grid_sample(source_alpha_2d[None, None], grid[None], mode="bilinear",
                                      padding_mode="zeros", align_corners=False)[0, 0]
        valid &= sampled_alpha >= min_alpha
    if sky_mask is not None:
        valid &= _depth_2d(sky_mask).to(device=target.device) < 0.5
    confidence = torch.where(
        valid, torch.exp(-relative / max(float(sigma_z), eps)), torch.zeros_like(relative))
    hard = valid & (relative <= float(relative_threshold))
    return DepthConsistencyResult(projected_z[None], sampled[None], relative[None], hard[None],
                                  confidence[None], grid)
