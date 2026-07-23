"""Per-view bounded exposure-interval camera trajectories for blurry views."""

from __future__ import annotations

import math

import torch
from torch import nn

from scene.pose_refinement import _bounded_vector, se3_exp


class PerViewBlurTrajectory(nn.Module):
    def __init__(self, blurred_mask: torch.Tensor, scene_radius: float,
                 max_rotation_deg: float = 0.30,
                 max_translation_radius_ratio: float = 0.003):
        super().__init__()
        mask = torch.as_tensor(blurred_mask, dtype=torch.bool).reshape(-1)
        if mask.numel() < 1 or scene_radius <= 0:
            raise ValueError("blur mask and scene radius must be non-empty/positive")
        self.register_buffer("blurred_mask", mask, persistent=True)
        self.max_rotation_radians = math.radians(float(max_rotation_deg))
        self.max_translation = float(scene_radius) * float(max_translation_radius_ratio)
        self.rotation_raw = nn.Parameter(torch.zeros((mask.numel(), 2, 3)))
        self.translation_raw = nn.Parameter(torch.zeros((mask.numel(), 2, 3)))

    @classmethod
    def from_cameras(cls, cameras: list, scene_radius: float, percentile: float = 0.30,
                     max_rotation_deg: float = 0.30,
                     max_translation_radius_ratio: float = 0.003):
        sharpness = torch.tensor([float(getattr(camera, "sharpness", 0.0)) for camera in cameras])
        threshold = torch.quantile(sharpness, float(percentile))
        return cls(sharpness <= threshold, scene_radius, max_rotation_deg,
                   max_translation_radius_ratio)

    def is_blurred(self, camera_index: int) -> bool:
        return bool(self.blurred_mask[int(camera_index)].item())

    def endpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (_bounded_vector(self.rotation_raw, self.max_rotation_radians),
                _bounded_vector(self.translation_raw, self.max_translation))

    def subposes(self, camera_index: int, num_samples: int) -> torch.Tensor:
        """Return column-vector correction matrices across the exposure interval."""

        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        index = int(camera_index)
        if not self.is_blurred(index):
            return torch.eye(4, device=self.rotation_raw.device,
                             dtype=self.rotation_raw.dtype)[None]
        rotation, translation = self.endpoints()
        times = torch.linspace(0.0, 1.0, num_samples, device=rotation.device, dtype=rotation.dtype)
        rotation_samples = ((1.0 - times[:, None]) * rotation[index, 0]
                            + times[:, None] * rotation[index, 1])
        translation_samples = ((1.0 - times[:, None]) * translation[index, 0]
                               + times[:, None] * translation[index, 1])
        return se3_exp(rotation_samples, translation_samples)

    def regularization(self) -> torch.Tensor:
        rotation, translation = self.endpoints()
        mask = self.blurred_mask
        if not mask.any():
            return (rotation.sum() + translation.sum()) * 0.0
        magnitude = rotation[mask].square().mean() + translation[mask].square().mean()
        interval = ((rotation[mask, 1] - rotation[mask, 0]).square().mean()
                    + (translation[mask, 1] - translation[mask, 0]).square().mean())
        return magnitude + interval

