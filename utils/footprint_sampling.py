"""Screen-space Gaussian ellipse projection and vectorized map sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.depth_reprojection import pixel_grid, project_world_points


def _covariance_matrix(covariance: torch.Tensor) -> torch.Tensor:
    if covariance.ndim == 3 and covariance.shape[-2:] == (3, 3):
        return covariance
    if covariance.ndim != 2 or covariance.shape[-1] != 6:
        raise ValueError("covariance must have shape [N,3,3] or packed shape [N,6]")
    result = covariance.new_zeros((covariance.shape[0], 3, 3))
    result[:, 0, 0] = covariance[:, 0]
    result[:, 0, 1] = result[:, 1, 0] = covariance[:, 1]
    result[:, 0, 2] = result[:, 2, 0] = covariance[:, 2]
    result[:, 1, 1] = covariance[:, 3]
    result[:, 1, 2] = result[:, 2, 1] = covariance[:, 4]
    result[:, 2, 2] = covariance[:, 5]
    return result


def project_gaussian_ellipse(
    xyz: torch.Tensor,
    covariance: torch.Tensor,
    camera,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D covariance to normalized ``grid_sample`` ellipse axes.

    Transforms use this repository's row-vector convention. Returned axes are
    two standard deviations in normalized grid coordinates. Depth used in the
    projection Jacobian is camera-space z-depth, not Euclidean ray depth.
    """

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    cov_world = _covariance_matrix(covariance).to(device=xyz.device, dtype=xyz.dtype)
    if cov_world.shape[0] != xyz.shape[0]:
        raise ValueError("covariance must have one row per Gaussian")

    u, v, z = project_world_points(xyz, camera)
    centers = pixel_grid(u, v, int(camera.image_width), int(camera.image_height))
    ones = torch.ones_like(xyz[:, :1])
    camera_points = torch.cat((xyz, ones), dim=-1) @ camera.world_view_transform.to(
        device=xyz.device, dtype=xyz.dtype)
    x, y = camera_points[:, 0], camera_points[:, 1]
    safe_z = z.clamp_min(torch.finfo(xyz.dtype).eps)
    jacobian = xyz.new_zeros((xyz.shape[0], 2, 3))
    jacobian[:, 0, 0] = float(camera.fx) / safe_z
    jacobian[:, 0, 2] = -float(camera.fx) * x / safe_z.square()
    jacobian[:, 1, 1] = float(camera.fy) / safe_z
    jacobian[:, 1, 2] = -float(camera.fy) * y / safe_z.square()

    world_to_camera = camera.world_view_transform[:3, :3].to(device=xyz.device, dtype=xyz.dtype)
    cov_camera = world_to_camera.transpose(0, 1)[None] @ cov_world @ world_to_camera[None]
    cov_screen = jacobian @ cov_camera @ jacobian.transpose(1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_screen)
    eigenvalues = eigenvalues.clamp_min(0.0)
    scale_to_grid = xyz.new_tensor((
        2.0 / float(camera.image_width), 2.0 / float(camera.image_height)))
    axis_minor = 2.0 * eigenvectors[:, :, 0] * torch.sqrt(eigenvalues[:, 0:1]) * scale_to_grid
    axis_major = 2.0 * eigenvectors[:, :, 1] * torch.sqrt(eigenvalues[:, 1:2]) * scale_to_grid
    valid = torch.isfinite(z) & (z > 0)
    centers = torch.nan_to_num(centers, nan=2.0, posinf=2.0, neginf=-2.0)
    axis_major = torch.where(valid[:, None], torch.nan_to_num(axis_major), torch.zeros_like(axis_major))
    axis_minor = torch.where(valid[:, None], torch.nan_to_num(axis_minor), torch.zeros_like(axis_minor))
    return centers, axis_major, axis_minor


def _map_2d(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    if value.ndim == 2:
        return value
    raise ValueError(f"{name} must have shape [H,W] or [1,H,W]")


def _pattern(major: torch.Tensor, minor: torch.Tensor, pattern: str) -> torch.Tensor:
    zero = torch.zeros_like(major)
    diagonal = 2.0 ** -0.5
    offsets = [zero, major, -major, minor, -minor,
               diagonal * (major + minor), diagonal * (major - minor),
               diagonal * (-major + minor), diagonal * (-major - minor)]
    if pattern == "13point":
        offsets += [1.5 * major, -1.5 * major, 1.5 * minor, -1.5 * minor]
    elif pattern != "9point":
        raise ValueError("pattern must be '9point' or '13point'")
    return torch.stack(offsets, dim=1)


def _masked_statistics(samples: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, ...]:
    count = valid.sum(dim=1)
    safe_count = count.clamp_min(1)
    mean = (samples * valid.to(samples.dtype)).sum(dim=1) / safe_count
    ordered = torch.sort(samples.masked_fill(~valid, float("inf")), dim=1).values
    p90_index = torch.floor(0.9 * (safe_count - 1).to(samples.dtype)).long()
    p90 = ordered.gather(1, p90_index[:, None]).squeeze(1)
    maximum = samples.masked_fill(~valid, float("-inf")).max(dim=1).values
    has_sample = count > 0
    return (torch.where(has_sample, mean, torch.zeros_like(mean)),
            torch.where(has_sample, p90, torch.zeros_like(p90)),
            torch.where(has_sample, maximum, torch.zeros_like(maximum)), count)


def sample_footprint_statistics(
    residual_map: torch.Tensor,
    edge_map: torch.Tensor,
    centers: torch.Tensor,
    axis_major: torch.Tensor,
    axis_minor: torch.Tensor,
    pattern: str = "9point",
    mean_weight: float = 0.50,
    p90_weight: float = 0.40,
    max_weight: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Sample residual/edge maps at a 9- or 13-point ellipse pattern."""

    residual = _map_2d(residual_map, "residual_map")
    edge = _map_2d(edge_map, "edge_map").to(device=residual.device, dtype=residual.dtype)
    if edge.shape != residual.shape:
        raise ValueError("residual_map and edge_map must have the same shape")
    if centers.shape != axis_major.shape or centers.shape != axis_minor.shape or centers.shape[-1] != 2:
        raise ValueError("centers and axes must all have shape [N,2]")
    offsets = _pattern(axis_major, axis_minor, pattern)
    points = centers[:, None] + offsets
    valid = (points[..., 0].abs() <= 1.0) & (points[..., 1].abs() <= 1.0)
    grid = points.reshape(1, -1, 1, 2)
    stacked = torch.stack((residual, edge), dim=0)[None]
    sampled = F.grid_sample(stacked, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)[0, :, :, 0].reshape(2, centers.shape[0], -1)
    r_mean, r_p90, r_max, count = _masked_statistics(sampled[0], valid)
    e_mean, e_p90, e_max, _ = _masked_statistics(sampled[1], valid)
    return {
        "residual_mean": r_mean,
        "residual_p90": r_p90,
        "residual_max": r_max,
        "residual_score": float(mean_weight) * r_mean + float(p90_weight) * r_p90 + float(max_weight) * r_max,
        "edge_mean": e_mean,
        "edge_p90": e_p90,
        "edge_max": e_max,
        "edge_score": float(mean_weight) * e_mean + float(p90_weight) * e_p90 + float(max_weight) * e_max,
        "valid_sample_count": count,
        "sampling_grid": points,
    }
