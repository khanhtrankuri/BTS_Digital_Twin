"""Low-order directional background that never contributes foreground depth."""

from __future__ import annotations

import torch
from torch import nn


class DirectionalSHBackground(nn.Module):
    """Per-scene RGB background evaluated from world-space ray direction.

    This uses a real SH-like polynomial basis through degree two. The module is
    deliberately low capacity and is composed only through ``1 - alpha``.
    """

    def __init__(self, degree: int = 2, initial_color=(0.0, 0.0, 0.0)):
        super().__init__()
        if degree not in (0, 1, 2):
            raise ValueError("Directional background degree must be 0, 1, or 2")
        self.degree = int(degree)
        coefficient_count = (self.degree + 1) ** 2
        coefficients = torch.zeros(coefficient_count, 3, dtype=torch.float32)
        color = torch.as_tensor(initial_color, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        coefficients[0] = torch.logit(color)
        self.coefficients = nn.Parameter(coefficients)

    def _basis(self, direction: torch.Tensor) -> torch.Tensor:
        x, y, z = direction.unbind(dim=-1)
        values = [torch.ones_like(x)]
        if self.degree >= 1:
            values += [x, y, z]
        if self.degree >= 2:
            values += [x * y, y * z, 3.0 * z.square() - 1.0, x * z, x.square() - y.square()]
        return torch.stack(values, dim=-1)

    def forward(self, camera) -> torch.Tensor:
        device, dtype = self.coefficients.device, self.coefficients.dtype
        height, width = int(camera.image_height), int(camera.image_width)
        v, u = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                              torch.arange(width, device=device, dtype=dtype), indexing="ij")
        x = (u + 0.5 - float(camera.cx)) / float(camera.fx)
        y = (v + 0.5 - float(camera.cy)) / float(camera.fy)
        rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
        rays = torch.nn.functional.normalize(rays, dim=-1, eps=1e-6)
        camera_to_world = torch.linalg.inv(camera.world_view_transform.to(device=device, dtype=dtype))
        world_rays = rays @ camera_to_world[:3, :3]
        world_rays = torch.nn.functional.normalize(world_rays, dim=-1, eps=1e-6)
        logits = self._basis(world_rays) @ self.coefficients
        return torch.sigmoid(logits).permute(2, 0, 1)
