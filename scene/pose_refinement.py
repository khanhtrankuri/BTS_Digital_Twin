"""Bounded train-camera SE(3) refinement for BTS-GeoGS Stage 1."""

from __future__ import annotations

import math

import torch
from torch import nn


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3))


def se3_exp(rotation_vector: torch.Tensor, translation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map returning column-vector SE(3) matrices."""

    if rotation_vector.shape != translation_vector.shape or rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector and translation_vector must share [...,3] shape")
    theta = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    theta2 = theta.square()
    small = theta < 1e-4
    safe_theta = theta.clamp_min(1e-8)
    safe_theta2 = theta2.clamp_min(1e-8)
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2.square() / 120.0,
                    torch.sin(theta) / safe_theta)
    b = torch.where(small, 0.5 - theta2 / 24.0 + theta2.square() / 720.0,
                    (1.0 - torch.cos(theta)) / safe_theta2)
    c = torch.where(small, 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
                    (theta - torch.sin(theta)) / (safe_theta2 * safe_theta))
    omega = _skew(rotation_vector)
    omega2 = omega @ omega
    identity = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    identity = identity.expand(rotation_vector.shape[:-1] + (3, 3))
    rotation = identity + a[..., None] * omega + b[..., None] * omega2
    v_matrix = identity + b[..., None] * omega + c[..., None] * omega2
    translated = (v_matrix @ translation_vector[..., None]).squeeze(-1)
    transform = torch.eye(4, device=rotation_vector.device, dtype=rotation_vector.dtype)
    transform = transform.expand(rotation_vector.shape[:-1] + (4, 4)).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translated
    return transform


def _bounded_vector(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = torch.tanh(norm) / norm.clamp_min(1e-8)
    return raw * scale * float(maximum)


class RefinedCamera:
    """Read-only camera proxy carrying differentiable corrected transforms."""

    def __init__(self, base_camera, world_view_transform: torch.Tensor):
        self._base_camera = base_camera
        self.world_view_transform = world_view_transform
        self.full_proj_transform = world_view_transform @ base_camera.projection_matrix
        self.camera_center = torch.linalg.inv(world_view_transform)[3, :3]

    def __getattr__(self, name):
        return getattr(self._base_camera, name)


class PerViewPoseRefinement(nn.Module):
    """Small bounded corrections for train cameras only; test cameras bypass it."""

    def __init__(self, num_cameras: int, scene_radius: float,
                 max_rotation_deg: float = 0.5,
                 max_translation_radius_ratio: float = 0.005):
        super().__init__()
        if num_cameras < 1 or scene_radius <= 0:
            raise ValueError("num_cameras and scene_radius must be positive")
        self.num_cameras = int(num_cameras)
        self.max_rotation_radians = math.radians(float(max_rotation_deg))
        self.max_translation = float(scene_radius) * float(max_translation_radius_ratio)
        self.rotation_raw = nn.Parameter(torch.zeros((num_cameras, 3)))
        self.translation_raw = nn.Parameter(torch.zeros((num_cameras, 3)))

    def corrections(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (_bounded_vector(self.rotation_raw, self.max_rotation_radians),
                _bounded_vector(self.translation_raw, self.max_translation))

    def correction_matrix(self, camera_index: int | torch.Tensor) -> torch.Tensor:
        rotation, translation = self.corrections()
        return se3_exp(rotation[camera_index], translation[camera_index])

    def refine_camera(self, camera) -> RefinedCamera:
        index = int(getattr(camera, "view_index", getattr(camera, "uid", -1)))
        if index < 0 or index >= self.num_cameras:
            raise IndexError(f"train camera index {index} is outside pose-refinement state")
        # Column convention: W2C' = Exp(xi) W2C. Repository matrices are the
        # transpose/row-vector form, hence W2C_row' = W2C_row Exp(xi)^T.
        correction_row = self.correction_matrix(index).transpose(-1, -2)
        return RefinedCamera(camera, camera.world_view_transform @ correction_row)

    def regularization(self, rotation_weight: float, translation_weight: float) -> torch.Tensor:
        rotation, translation = self.corrections()
        return (float(rotation_weight) * rotation.square().sum(dim=-1).mean()
                + float(translation_weight) * translation.square().sum(dim=-1).mean())

    def trajectory_smoothness(self) -> torch.Tensor:
        rotation, translation = self.corrections()
        if self.num_cameras < 3:
            return (rotation.sum() + translation.sum()) * 0.0
        rotation_second = rotation[:-2] - 2.0 * rotation[1:-1] + rotation[2:]
        translation_second = translation[:-2] - 2.0 * translation[1:-1] + translation[2:]
        return rotation_second.square().mean() + translation_second.square().mean()

    @torch.no_grad()
    def diagnostics(self, scene_radius: float) -> dict[str, float]:
        rotation, translation = self.corrections()
        rotation_degrees = torch.linalg.vector_norm(rotation, dim=-1) * (180.0 / math.pi)
        translation_ratio = torch.linalg.vector_norm(translation, dim=-1) / max(float(scene_radius), 1e-8)
        return {
            "rotation_mean_degrees": float(rotation_degrees.mean().item()),
            "rotation_max_degrees": float(rotation_degrees.max().item()),
            "translation_mean_radius_ratio": float(translation_ratio.mean().item()),
            "translation_max_radius_ratio": float(translation_ratio.max().item()),
        }

