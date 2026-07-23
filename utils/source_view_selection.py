"""Geometry, sharpness, time and exposure-aware Stage-1 source selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from utils.depth_reprojection import pixel_grid, project_world_points, unproject_pixels


@dataclass(frozen=True)
class SourceViewScore:
    total: float
    position: float
    angle: float
    overlap: float
    sharpness: float
    temporal: float
    exposure_mismatch: float


def _cfg(cfg, name: str, default):
    return getattr(cfg, name, default)


def camera_forward(camera) -> torch.Tensor:
    """Return the world-space +z optical axis under the row-vector convention."""

    camera_to_world = torch.linalg.inv(camera.world_view_transform)
    return F.normalize(camera_to_world[2, :3], dim=0, eps=1e-8)


def estimate_frustum_overlap(target_camera, source_camera, scene_radius: float) -> float:
    """Estimate overlap by projecting a 3x3 target-frustum proxy plane."""

    device = target_camera.world_view_transform.device
    dtype = target_camera.world_view_transform.dtype
    u_values = torch.tensor(
        [0.0, 0.5 * (target_camera.image_width - 1), target_camera.image_width - 1.0],
        device=device, dtype=dtype)
    v_values = torch.tensor(
        [0.0, 0.5 * (target_camera.image_height - 1), target_camera.image_height - 1.0],
        device=device, dtype=dtype)
    v, u = torch.meshgrid(v_values, u_values, indexing="ij")
    proxy_depth = torch.full_like(u, max(float(scene_radius), 1e-4))
    world = unproject_pixels(u, v, proxy_depth, target_camera)
    projected_u, projected_v, projected_z = project_world_points(world, source_camera)
    grid = pixel_grid(projected_u, projected_v, source_camera.image_width, source_camera.image_height)
    inside = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0) & (projected_z > 0)
    return float(inside.float().mean().item())


def _is_valid_camera(camera) -> bool:
    values = (getattr(camera, "fx", 0.0), getattr(camera, "fy", 0.0),
              getattr(camera, "image_width", 0), getattr(camera, "image_height", 0))
    return (bool(getattr(camera, "has_ground_truth", True))
            and all(float(value) > 0 for value in values)
            and torch.isfinite(camera.camera_center).all().item())


def score_source_view(target_camera, candidate_camera, scene_radius: float, cfg,
                      temporal_scale: float = 1.0) -> SourceViewScore:
    radius = max(float(scene_radius), 1e-6)
    distance = torch.linalg.vector_norm(candidate_camera.camera_center - target_camera.camera_center)
    position = float(torch.exp(-distance / radius).item())
    cosine = torch.dot(camera_forward(target_camera), camera_forward(candidate_camera)).clamp(-1.0, 1.0)
    angle = float((0.5 * (cosine + 1.0)).item())
    overlap = estimate_frustum_overlap(target_camera, candidate_camera, radius)
    sharpness = float(max(0.0, min(1.0, getattr(candidate_camera, "normalized_sharpness", 1.0))))

    target_time = getattr(target_camera, "frame_time", None)
    candidate_time = getattr(candidate_camera, "frame_time", None)
    if target_time is None or candidate_time is None:
        temporal = 0.0
    else:
        temporal = math.exp(
            -abs(float(candidate_time) - float(target_time)) / max(float(temporal_scale), 1e-6))
    target_brightness = getattr(target_camera, "mean_brightness", None)
    candidate_brightness = getattr(candidate_camera, "mean_brightness", None)
    exposure_mismatch = (abs(float(candidate_brightness) - float(target_brightness))
                         if target_brightness is not None and candidate_brightness is not None else 0.0)
    total = (
        float(_cfg(cfg, "multiview_position_weight", 1.0)) * position
        + float(_cfg(cfg, "multiview_angle_weight", 1.0)) * angle
        + float(_cfg(cfg, "multiview_overlap_weight", 1.0)) * overlap
        + float(_cfg(cfg, "multiview_sharpness_weight", 0.5)) * sharpness
        + float(_cfg(cfg, "multiview_temporal_weight", 0.0)) * temporal
        - float(_cfg(cfg, "multiview_exposure_mismatch_weight", 0.25)) * exposure_mismatch
    )
    return SourceViewScore(total, position, angle, overlap, sharpness, temporal, exposure_mismatch)


def select_source_views(
    target_camera,
    candidate_cameras: list,
    num_sources: int,
    scene_radius: float,
    cfg,
) -> list:
    """Return high-scoring valid overlapping source cameras, never the target."""

    candidates = [camera for camera in candidate_cameras
                  if camera is not target_camera
                  and getattr(camera, "image_name", None) != getattr(target_camera, "image_name", None)
                  and _is_valid_camera(camera)]
    if not candidates or num_sources <= 0:
        return []
    target_time = getattr(target_camera, "frame_time", None)
    time_differences = [abs(float(camera.frame_time) - float(target_time)) for camera in candidates
                        if target_time is not None and getattr(camera, "frame_time", None) is not None]
    temporal_scale = max(time_differences) if time_differences else 1.0
    scored = [(camera, score_source_view(target_camera, camera, scene_radius, cfg, temporal_scale))
              for camera in candidates]
    minimum_overlap = float(_cfg(cfg, "multiview_min_overlap", 0.20))
    scored = [item for item in scored if item[1].overlap >= minimum_overlap]
    # Reject a blurred duplicate when a materially sharper camera has nearly the
    # same geometric utility. We keep isolated blurry views rather than losing
    # coverage in sparse regions.
    blur_threshold = float(_cfg(cfg, "multiview_blur_source_threshold", 0.35))
    equivalence_margin = float(_cfg(cfg, "multiview_sharp_equivalence_margin", 0.10))
    filtered = []
    for camera, score in scored:
        has_sharp_equivalent = score.sharpness < blur_threshold and any(
            other_score.sharpness >= blur_threshold
            and other_score.sharpness >= score.sharpness + 0.20
            and abs(other_score.position - score.position) <= equivalence_margin
            and abs(other_score.angle - score.angle) <= equivalence_margin
            and abs(other_score.overlap - score.overlap) <= equivalence_margin
            for other_camera, other_score in scored if other_camera is not camera)
        if not has_sharp_equivalent:
            filtered.append((camera, score))
    scored = filtered
    scored.sort(key=lambda item: item[1].total, reverse=True)
    if not scored:
        return []

    selected = []
    if bool(_cfg(cfg, "multiview_temporal_bracketing", False)) and target_time is not None and num_sources >= 2:
        before = [item for item in scored if getattr(item[0], "frame_time", None) is not None
                  and float(item[0].frame_time) < float(target_time)]
        after = [item for item in scored if getattr(item[0], "frame_time", None) is not None
                 and float(item[0].frame_time) > float(target_time)]
        if before:
            selected.append(before[0][0])
        if after:
            selected.append(after[0][0])
    for camera, _ in scored:
        if camera not in selected:
            selected.append(camera)
        if len(selected) >= int(num_sources):
            break
    return selected[:int(num_sources)]
