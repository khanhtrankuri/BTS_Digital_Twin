"""NAFNet building blocks.

The implementation follows the activation-free NAFNet design while keeping
the normalization implementation portable across Linux, Windows and AMP.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for ``[B,C,H,W]`` tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.weight.numel():
            raise ValueError(
                f"LayerNorm2d expected [B,{self.weight.numel()},H,W], "
                f"got {tuple(value.shape)}"
            )
        normalized = F.layer_norm(
            value.permute(0, 2, 3, 1),
            (self.weight.numel(),),
            self.weight,
            self.bias,
            self.eps,
        )
        return normalized.permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    """Split channels in half and multiply both halves."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] % 2:
            raise ValueError("SimpleGate requires an even channel count")
        first, second = value.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Activation-free residual block used by NAFNet."""

    def __init__(
        self,
        channels: int,
        depthwise_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        depth_channels = channels * int(depthwise_expand)
        ffn_channels = channels * int(ffn_expand)
        if depth_channels % 2 or ffn_channels % 2:
            raise ValueError("Expanded NAFBlock channel counts must be even")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depth_channels, 1)
        self.conv2 = nn.Conv2d(
            depth_channels,
            depth_channels,
            3,
            padding=1,
            groups=depth_channels,
        )
        self.gate1 = SimpleGate()
        gated_channels = depth_channels // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1),
        )
        self.conv3 = nn.Conv2d(gated_channels, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)

        self.dropout1 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv1(self.norm1(value))
        value = self.gate1(self.conv2(value))
        value = value * self.channel_attention(value)
        value = self.dropout1(self.conv3(value))
        value = residual + value * self.beta

        residual = value
        value = self.conv4(self.norm2(value))
        value = self.dropout2(self.conv5(self.gate2(value)))
        return residual + value * self.gamma
