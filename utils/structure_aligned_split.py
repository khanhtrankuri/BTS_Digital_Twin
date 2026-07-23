"""Structure-aligned Gaussian splitting for BTS-GeoGS Stage 1.

Input scaling is activated (positive) XYZ-local standard deviation. Quaternion
rotation follows ``utils.general_utils.build_rotation``: covariance axes are
the columns of the returned local-to-world rotation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.densification_utils import split_opacity_conserving
from utils.general_utils import build_rotation


@dataclass
class SplitResult:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    raw_opacity: torch.Tensor
    shape_type: torch.Tensor
    parent_distance: torch.Tensor


def classify_gaussian_shape(
    scaling: torch.Tensor,
    wire_ratio: float,
    surface_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify positive Gaussian scales as disjoint wire/surface/blob masks."""

    if scaling.ndim != 2 or scaling.shape[-1] != 3:
        raise ValueError("scaling must have shape [N,3]")
    if wire_ratio <= 1.0 or surface_ratio <= 1.0:
        raise ValueError("shape ratio thresholds must be greater than one")
    ordered = torch.sort(scaling.clamp_min(torch.finfo(scaling.dtype).tiny), dim=-1,
                         descending=True).values
    wire = ordered[:, 0] / ordered[:, 1] > float(wire_ratio)
    surface = (~wire) & (ordered[:, 1] / ordered[:, 2] > float(surface_ratio))
    blob = ~(wire | surface)
    return wire, surface, blob


def _cfg(cfg, name: str, default):
    return getattr(cfg, name, default)


def _axis_factors(order: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(factors).scatter(1, order, factors)


def structure_aligned_split(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    raw_opacity: torch.Tensor,
    error_direction: torch.Tensor | None,
    cfg,
    num_outputs: int = 2,
) -> SplitResult:
    """Split wires axially and surfaces tangentially; blobs retain randomness.

    ``error_direction`` is an optional world-space direction per parent. Its
    normal component is removed for surface Gaussians. Output ordering matches
    the legacy ``tensor.repeat(num_outputs, ...)`` convention.
    """

    if num_outputs < 2:
        raise ValueError("structure-aligned splitting needs at least two outputs")
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or scaling.shape != xyz.shape:
        raise ValueError("xyz and scaling must both have shape [N,3]")
    if rotation.ndim != 2 or rotation.shape != (xyz.shape[0], 4):
        raise ValueError("rotation must have shape [N,4]")
    if raw_opacity.shape[0] != xyz.shape[0]:
        raise ValueError("raw_opacity must have one row per Gaussian")
    if error_direction is not None and error_direction.shape != xyz.shape:
        raise ValueError("error_direction must have shape [N,3]")

    wire, surface, blob = classify_gaussian_shape(
        scaling,
        float(_cfg(cfg, "wire_ratio_threshold", 4.0)),
        float(_cfg(cfg, "surface_ratio_threshold", 4.0)),
    )
    ordered_scales, order = torch.sort(scaling, dim=-1, descending=True)
    rotation_matrix = build_rotation(rotation)
    gather_index = order[:, None, :].expand(-1, 3, -1)
    axes_sorted = torch.gather(rotation_matrix, 2, gather_index)
    major_axis, normal_axis = axes_sorted[:, :, 0], axes_sorted[:, :, 2]

    direction = major_axis
    if error_direction is not None:
        projected = error_direction - (error_direction * normal_axis).sum(-1, keepdim=True) * normal_axis
        valid_projected = torch.linalg.vector_norm(projected, dim=-1, keepdim=True) > 1e-8
        tangent = F.normalize(projected, dim=-1, eps=1e-8)
        direction = torch.where(valid_projected, tangent, major_axis)

    wire_offset = float(_cfg(cfg, "wire_split_offset", 0.35)) * ordered_scales[:, :1]
    surface_offset = float(_cfg(cfg, "surface_split_offset", 0.30)) * ordered_scales[:, :1]
    deterministic_offset = torch.zeros_like(xyz)
    deterministic_offset[wire] = major_axis[wire] * wire_offset[wire]
    deterministic_offset[surface] = direction[surface] * surface_offset[surface]

    signs = torch.linspace(-1.0, 1.0, num_outputs, device=xyz.device, dtype=xyz.dtype)
    offsets = signs[:, None, None] * deterministic_offset[None]
    if blob.any():
        if bool(_cfg(cfg, "blob_random_split", True)):
            samples = torch.normal(
                mean=torch.zeros((num_outputs, xyz.shape[0], 3), device=xyz.device, dtype=xyz.dtype),
                std=scaling[None].expand(num_outputs, -1, -1),
            )
            random_world = torch.einsum("nij,knj->kni", rotation_matrix, samples)
            offsets[:, blob] = random_world[:, blob]
        else:
            blob_offset = major_axis[blob] * (0.30 * ordered_scales[blob, :1])
            offsets[:, blob] = signs[:, None, None] * blob_offset[None]

    child_scaling = scaling / (0.8 * float(num_outputs))
    if wire.any():
        wire_factors_sorted = torch.stack((
            torch.full_like(ordered_scales[:, 0], float(_cfg(cfg, "wire_major_scale_factor", 0.58))),
            torch.full_like(ordered_scales[:, 1], float(_cfg(cfg, "wire_minor_scale_factor", 0.85))),
            torch.full_like(ordered_scales[:, 2], float(_cfg(cfg, "wire_minor_scale_factor", 0.85))),
        ), dim=-1)
        wire_scaling = scaling * _axis_factors(order, wire_factors_sorted)
        child_scaling = torch.where(wire[:, None], wire_scaling, child_scaling)

    child_opacity = (split_opacity_conserving(raw_opacity, num_outputs)
                     if bool(_cfg(cfg, "opacity_conserving_split", False)) else raw_opacity)
    shape_type = torch.full((xyz.shape[0],), 2, device=xyz.device, dtype=torch.long)
    shape_type[wire], shape_type[surface] = 0, 1
    flat_offsets = offsets.reshape(-1, 3)
    return SplitResult(
        xyz=(xyz[None] + offsets).reshape(-1, 3),
        scaling=child_scaling.repeat(num_outputs, 1),
        rotation=rotation.repeat(num_outputs, 1),
        raw_opacity=child_opacity.repeat(num_outputs, 1),
        shape_type=shape_type.repeat(num_outputs),
        parent_distance=torch.linalg.vector_norm(flat_offsets, dim=-1),
    )
