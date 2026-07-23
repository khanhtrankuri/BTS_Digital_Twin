"""Full-resolution residual/edge/random patch sampling and crop intrinsics."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.graphics_utils import getProjectionMatrix


@dataclass(frozen=True)
class PatchCrop:
    left: int
    top: int
    width: int
    height: int
    category: str

    @property
    def slices(self):
        return (..., slice(self.top, self.top + self.height),
                slice(self.left, self.left + self.width))


class CroppedCamera:
    """Camera proxy with principal point shifted into a pixel crop."""

    def __init__(self, camera, crop: PatchCrop):
        self._base_camera = camera
        self.image_width = self.width = int(crop.width)
        self.image_height = self.height = int(crop.height)
        self.fx, self.fy = float(camera.fx), float(camera.fy)
        self.cx, self.cy = float(camera.cx) - crop.left, float(camera.cy) - crop.top
        self.projection_matrix = getProjectionMatrix(
            znear=camera.znear, zfar=camera.zfar,
            fovX=camera.FoVx, fovY=camera.FoVy,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            width=self.image_width, height=self.image_height,
        ).transpose(0, 1).to(device=camera.world_view_transform.device,
                            dtype=camera.world_view_transform.dtype)
        self.world_view_transform = camera.world_view_transform
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = camera.camera_center

    def __getattr__(self, name):
        return getattr(self._base_camera, name)


def _allowed_patch_positions(
    score: torch.Tensor,
    patch_size: int,
    previous_centers: list[tuple[float, float]] | None,
    min_patch_distance: float,
) -> torch.Tensor:
    height = score.shape[-2] - patch_size + 1
    width = score.shape[-1] - patch_size + 1
    allowed = torch.ones((height, width), dtype=torch.bool, device=score.device)
    if not previous_centers or min_patch_distance <= 0:
        return allowed
    top = torch.arange(height, device=score.device, dtype=score.dtype)[:, None]
    left = torch.arange(width, device=score.device, dtype=score.dtype)[None, :]
    center_offset = 0.5 * float(patch_size - 1)
    center_y, center_x = top + center_offset, left + center_offset
    minimum_squared = float(min_patch_distance) ** 2
    for previous_y, previous_x in previous_centers:
        allowed &= ((center_y - float(previous_y)).square()
                    + (center_x - float(previous_x)).square()) >= minimum_squared
    # Small images or long histories can exhaust the constraint. Falling back to
    # all positions keeps the training step well-defined.
    return allowed if allowed.any() else torch.ones_like(allowed)


def _best_patch_center(score: torch.Tensor, patch_size: int,
                       allowed: torch.Tensor) -> tuple[int, int]:
    pooled = F.avg_pool2d(score[None, None], kernel_size=patch_size, stride=1)[0, 0]
    pooled = pooled.masked_fill(~allowed, -torch.inf)
    flat = int(torch.argmax(pooled).item())
    width = pooled.shape[1]
    return flat // width, flat % width


def sample_patch(residual_map: torch.Tensor, edge_map: torch.Tensor, patch_size: int,
                 random_ratio: float = 0.50, residual_ratio: float = 0.30,
                 edge_ratio: float = 0.20,
                 thin_structure_map: torch.Tensor | None = None,
                 previous_centers: list[tuple[float, float]] | None = None,
                 min_patch_distance: float = 0.0) -> PatchCrop:
    """Sample a separated random/high-residual/high-edge-or-thin full-res patch."""

    residual = residual_map[0] if residual_map.ndim == 3 else residual_map
    edge = edge_map[0] if edge_map.ndim == 3 else edge_map
    if residual.ndim != 2 or residual.shape != edge.shape:
        raise ValueError("residual_map and edge_map must share [H,W] shape")
    if thin_structure_map is not None:
        thin = thin_structure_map[0] if thin_structure_map.ndim == 3 else thin_structure_map
        if thin.shape != edge.shape:
            raise ValueError("thin_structure_map must share residual spatial shape")
        edge = torch.maximum(edge, thin.to(device=edge.device, dtype=edge.dtype))
    height, width = residual.shape
    size = min(int(patch_size), height, width)
    if (min(random_ratio, residual_ratio, edge_ratio) < 0
            or abs(random_ratio + residual_ratio + edge_ratio - 1.0) > 1e-8):
        raise ValueError("patch sampling ratios must be non-negative and sum to one")
    allowed = _allowed_patch_positions(
        residual, size, previous_centers, float(min_patch_distance))
    draw = random.random()
    if draw < random_ratio:
        category = "random"
        choices = torch.nonzero(allowed, as_tuple=False)
        selected = choices[random.randrange(choices.shape[0])]
        top, left = int(selected[0].item()), int(selected[1].item())
    elif draw < random_ratio + residual_ratio:
        category = "residual"
        top, left = _best_patch_center(torch.nan_to_num(residual), size, allowed)
    else:
        category = "edge"
        top, left = _best_patch_center(torch.nan_to_num(edge), size, allowed)
    return PatchCrop(int(left), int(top), size, size, category)


def crop_camera(camera, crop: PatchCrop) -> CroppedCamera:
    return CroppedCamera(camera, crop)
