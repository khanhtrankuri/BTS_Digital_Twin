"""Bounded per-view exposure compensation for BTS-GeoGS-v3."""

from __future__ import annotations

import math
import re

import torch
from torch import nn


class PerViewExposure(nn.Module):
    """Diagonal RGB gain and bias with differentiable hard bounds.

    Gain is parameterized with a shifted/scaled tanh so identity is represented
    exactly at initialization even when the configured bounds are asymmetric.
    Bias uses the requested bounded tanh parameterization.
    """

    def __init__(self, num_views: int, min_gain: float = 0.75,
                 max_gain: float = 1.25, max_bias: float = 0.10):
        super().__init__()
        if num_views < 0:
            raise ValueError("num_views must be non-negative")
        if not (0.0 < min_gain <= 1.0 <= max_gain):
            raise ValueError("Exposure gain bounds must satisfy 0 < min_gain <= 1 <= max_gain")
        if max_gain <= min_gain:
            raise ValueError("max_gain must be greater than min_gain")
        if max_bias < 0:
            raise ValueError("max_bias must be non-negative")

        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.max_bias = float(max_bias)
        midpoint = 0.5 * (self.min_gain + self.max_gain)
        half_range = 0.5 * (self.max_gain - self.min_gain)
        identity_normalized = max(-1.0 + 1e-7, min(1.0 - 1e-7, (1.0 - midpoint) / half_range))
        identity_raw = 0.5 * math.log((1.0 + identity_normalized) / (1.0 - identity_normalized))

        self.raw_gain = nn.Parameter(torch.full((num_views, 3), identity_raw, dtype=torch.float32))
        self.raw_bias = nn.Parameter(torch.zeros((num_views, 3), dtype=torch.float32))
        self.register_buffer("camera_positions", torch.empty((0, 3), dtype=torch.float32), persistent=True)
        self.register_buffer("camera_directions", torch.empty((0, 3), dtype=torch.float32), persistent=True)
        self.register_buffer("camera_focals", torch.empty((0,), dtype=torch.float32), persistent=True)
        self.register_buffer("camera_times", torch.empty((0,), dtype=torch.float32), persistent=True)

    @property
    def num_views(self) -> int:
        return int(self.raw_gain.shape[0])

    def gains(self) -> torch.Tensor:
        midpoint = 0.5 * (self.min_gain + self.max_gain)
        half_range = 0.5 * (self.max_gain - self.min_gain)
        return midpoint + half_range * torch.tanh(self.raw_gain)

    def biases(self) -> torch.Tensor:
        return self.max_bias * torch.tanh(self.raw_bias)

    def get_gain_bias(self, view_index):
        if self.num_views == 0:
            device = self.raw_gain.device
            return torch.ones(3, device=device), torch.zeros(3, device=device)
        index = torch.as_tensor(view_index, dtype=torch.long, device=self.raw_gain.device)
        return self.gains()[index], self.biases()[index]

    def forward(self, image: torch.Tensor, view_index) -> torch.Tensor:
        gain, bias = self.get_gain_bias(view_index)
        while gain.ndim < image.ndim:
            gain = gain.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        return gain * image + bias

    def identity(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def regularization_loss(self, gain_weight: float = 0.001,
                            bias_weight: float = 0.001,
                            zero_mean_weight: float = 0.0) -> torch.Tensor:
        if self.num_views == 0:
            return self.raw_gain.sum() * 0.0
        gain, bias = self.gains(), self.biases()
        loss = float(gain_weight) * (gain - 1.0).square().mean()
        loss = loss + float(bias_weight) * bias.square().mean()
        if zero_mean_weight:
            mean_loss = (gain - 1.0).mean(dim=0).square().mean() + bias.mean(dim=0).square().mean()
            loss = loss + float(zero_mean_weight) * mean_loss
        return loss

    def as_matrices(self) -> torch.Tensor:
        matrices = self.raw_gain.new_zeros((self.num_views, 3, 4))
        if self.num_views:
            matrices[:, 0, 0] = self.gains()[:, 0]
            matrices[:, 1, 1] = self.gains()[:, 1]
            matrices[:, 2, 2] = self.gains()[:, 2]
            matrices[:, :, 3] = self.biases()
        return matrices

    @torch.no_grad()
    def load_gain_bias(self, gain: torch.Tensor, bias: torch.Tensor) -> None:
        gain = torch.as_tensor(gain, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
        bias = torch.as_tensor(bias, dtype=self.raw_bias.dtype, device=self.raw_bias.device)
        if gain.shape != self.raw_gain.shape or bias.shape != self.raw_bias.shape:
            raise ValueError("Exposure state shape does not match the current training camera count")
        midpoint = 0.5 * (self.min_gain + self.max_gain)
        half_range = 0.5 * (self.max_gain - self.min_gain)
        normalized = ((gain.clamp(self.min_gain, self.max_gain) - midpoint) / half_range).clamp(-1 + 1e-6, 1 - 1e-6)
        self.raw_gain.copy_(torch.atanh(normalized))
        if self.max_bias > 0:
            self.raw_bias.copy_(torch.atanh((bias.clamp(-self.max_bias, self.max_bias) / self.max_bias).clamp(-1 + 1e-6, 1 - 1e-6)))
        else:
            self.raw_bias.zero_()

    @torch.no_grad()
    def set_camera_poses(self, positions, directions, focals=None, times=None) -> None:
        positions = torch.as_tensor(positions, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
        directions = torch.as_tensor(directions, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
        if positions.shape != (self.num_views, 3) or directions.shape != (self.num_views, 3):
            raise ValueError("Camera pose arrays must have shape [num_views, 3]")
        self.camera_positions = positions
        self.camera_directions = torch.nn.functional.normalize(directions, dim=-1, eps=1e-6)

        def optional_vector(value, name):
            if value is None:
                return torch.empty((0,), dtype=self.raw_gain.dtype, device=self.raw_gain.device)
            result = torch.as_tensor(value, dtype=self.raw_gain.dtype, device=self.raw_gain.device).reshape(-1)
            if result.shape != (self.num_views,):
                raise ValueError(f"{name} must have shape [num_views]")
            return result

        self.camera_focals = optional_vector(focals, "focals")
        self.camera_times = optional_vector(times, "times")

    def infer_gain_bias(self, position: torch.Tensor, direction: torch.Tensor,
                        mode: str = "identity", k: int = 4, *, focal=None, time=None,
                        position_weight: float = 1.0, angle_weight: float = 1.0,
                        temporal_weight: float = 0.0, focal_weight: float = 0.0,
                        distance_temperature: float = 0.10,
                        confidence_temperature: float = 0.08,
                        min_confidence: float = 0.0,
                        max_gain_delta: float | None = None,
                        max_bias: float | None = None,
                        return_diagnostics: bool = False):
        """Infer bounded test-view exposure from pose and optional time.

        ``temporal_weighted`` uses only frame time when it is available;
        ``pose_temporal_weighted`` combines all configured terms; and
        ``pose_confidence_blend`` additionally fades the estimate to identity
        when the nearest source is far away.
        """
        if mode == "identity" or self.num_views == 0 or self.camera_positions.shape[0] != self.num_views:
            result = (self.raw_gain.new_ones(3), self.raw_gain.new_zeros(3))
            return (*result, {"mode": "identity", "confidence": 0.0}) if return_diagnostics else result
        supported = {"nearest_camera", "weighted_nearest", "pose_confidence_blend",
                     "temporal_weighted", "pose_temporal_weighted"}
        if mode not in supported:
            raise ValueError(f"Unsupported test exposure mode: {mode}")
        position = torch.as_tensor(position, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
        direction = torch.nn.functional.normalize(
            torch.as_tensor(direction, dtype=self.raw_gain.dtype, device=self.raw_gain.device), dim=-1, eps=1e-6)
        scene_scale = torch.linalg.vector_norm(
            self.camera_positions - self.camera_positions.mean(dim=0), dim=-1).median().clamp_min(1e-6)
        position_distance = torch.linalg.vector_norm(self.camera_positions - position, dim=-1) / scene_scale
        direction_distance = torch.acos(
            (self.camera_directions * direction).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)) / math.pi
        focal_distance = torch.zeros_like(position_distance)
        if focal is not None and self.camera_focals.shape[0] == self.num_views:
            query_focal = torch.as_tensor(focal, dtype=self.raw_gain.dtype, device=self.raw_gain.device).clamp_min(1e-6)
            focal_distance = torch.abs(torch.log(self.camera_focals.clamp_min(1e-6) / query_focal))
        time_distance = torch.zeros_like(position_distance)
        time_available = time is not None and self.camera_times.shape[0] == self.num_views
        if time_available:
            query_time = torch.as_tensor(time, dtype=self.raw_gain.dtype, device=self.raw_gain.device)
            time_scale = (self.camera_times.max() - self.camera_times.min()).clamp_min(1.0)
            time_distance = torch.abs(self.camera_times - query_time) / time_scale

        if mode == "temporal_weighted" and time_available:
            distance = time_distance
        else:
            use_temporal = mode == "pose_temporal_weighted" or float(temporal_weight) > 0.0
            distance = (float(position_weight) * position_distance
                        + float(angle_weight) * direction_distance
                        + float(focal_weight) * focal_distance
                        + (float(temporal_weight) * time_distance if use_temporal and time_available else 0.0))
        gains, biases = self.gains(), self.biases()
        if mode == "nearest_camera":
            index = torch.argmin(distance)
            gain, bias = gains[index], biases[index]
            diagnostics = {"mode": mode, "indices": [int(index.item())], "weights": [1.0],
                           "nearest_distance": float(distance[index].item()), "confidence": 1.0}
            return (gain, bias, diagnostics) if return_diagnostics else (gain, bias)
        count = min(max(1, int(k)), self.num_views)
        values, indices = torch.topk(distance, k=count, largest=False)
        temperature = max(float(distance_temperature), 1e-6)
        weights = torch.exp(-(values - values.min()) / temperature)
        weights = weights / weights.sum()
        gain = (weights[:, None] * gains[indices]).sum(dim=0)
        bias = (weights[:, None] * biases[indices]).sum(dim=0)
        confidence = torch.exp(-values[0] / max(float(confidence_temperature), 1e-6))
        confidence = torch.where(confidence >= float(min_confidence), confidence, torch.zeros_like(confidence))
        if mode == "pose_confidence_blend":
            gain = 1.0 + confidence * (gain - 1.0)
            bias = confidence * bias
        if max_gain_delta is not None:
            delta = abs(float(max_gain_delta))
            gain = gain.clamp(1.0 - delta, 1.0 + delta)
        if max_bias is not None:
            bias = bias.clamp(-abs(float(max_bias)), abs(float(max_bias)))
        diagnostics = {
            "mode": mode, "indices": [int(v) for v in indices.detach().cpu().tolist()],
            "weights": [float(v) for v in weights.detach().cpu().tolist()],
            "distances": [float(v) for v in values.detach().cpu().tolist()],
            "nearest_distance": float(values[0].item()), "confidence": float(confidence.item()),
            "gain": [float(v) for v in gain.detach().cpu().tolist()],
            "bias": [float(v) for v in bias.detach().cpu().tolist()],
        }
        return (gain, bias, diagnostics) if return_diagnostics else (gain, bias)


def frame_time_from_name(image_name: str) -> float | None:
    """Extract the final numeric token from an image name as capture time."""

    tokens = re.findall(r"\d+", str(image_name))
    return float(tokens[-1]) if tokens else None
