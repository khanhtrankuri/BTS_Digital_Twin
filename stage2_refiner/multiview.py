"""Verified inverse-warp primitives for optional future multi-view regularization."""

import torch
import torch.nn.functional as F


def inverse_warp(source, target_depth, source_intrinsics, target_intrinsics,
                 source_w2c, target_w2c, source_depth=None, alpha=None,
                 depth_consistency_threshold=0.02):
    """Sample ``source`` at target pixels using target depth and camera matrices."""
    batch, _, height, width = target_depth.shape
    ys, xs = torch.meshgrid(torch.arange(height, device=source.device, dtype=source.dtype),
                            torch.arange(width, device=source.device, dtype=source.dtype), indexing="ij")
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=0).reshape(1, 3, -1).repeat(batch, 1, 1)
    target_points = torch.linalg.inv(target_intrinsics) @ pixels * target_depth.reshape(batch, 1, -1)
    target_h = torch.cat((target_points, torch.ones(batch, 1, target_points.shape[-1], device=source.device, dtype=source.dtype)), dim=1)
    world = torch.linalg.inv(target_w2c) @ target_h
    source_points = source_w2c @ world
    projected = source_intrinsics @ source_points[:, :3]
    z = projected[:, 2:3]
    xy = projected[:, :2] / z.clamp_min(1e-8)
    grid_x = 2.0 * xy[:, 0] / max(1, width - 1) - 1.0
    grid_y = 2.0 * xy[:, 1] / max(1, height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(batch, height, width, 2)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (z.reshape(batch, 1, height, width) > 0) & (grid[..., 0:1].permute(0,3,1,2).abs() <= 1) & (grid[..., 1:2].permute(0,3,1,2).abs() <= 1)
    if alpha is not None: valid &= F.grid_sample(alpha, grid, align_corners=True) > 0.01
    if source_depth is not None:
        sampled_depth = F.grid_sample(source_depth, grid, align_corners=True)
        expected = source_points[:, 2:3].reshape(batch, 1, height, width)
        valid &= (sampled_depth - expected).abs() <= depth_consistency_threshold * expected.abs().clamp_min(1e-6)
    return warped, valid


def masked_multiview_l1(warped, target, valid):
    if not valid.any(): return warped.sum() * 0.0
    return ((warped - target).abs() * valid).sum() / (valid.sum() * warped.shape[1]).clamp_min(1)
