"""Configurable NAFNet encoder-decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .nafblock import NAFBlock


def _blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(NAFBlock(channels) for _ in range(int(count))))


class NAFNet(nn.Module):
    """NAFNet backbone producing unconstrained output channels.

    Spatial padding is internal and removed before returning, so arbitrary
    input resolutions are supported without changing camera/image geometry.
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 4,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        dec_blk_nums: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Encoder and decoder stage counts must match")
        if min(in_channels, out_channels, width) <= 0:
            raise ValueError("NAFNet channel counts must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.padder_size = 2 ** len(enc_blk_nums)
        self.intro = nn.Conv2d(self.in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, self.out_channels, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for count in enc_blk_nums:
            self.encoders.append(_blocks(channels, count))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = _blocks(channels, middle_blk_num)

        for count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(_blocks(channels, count))

        # An untrained refiner starts as an exact identity through delta=0.
        nn.init.zeros_(self.ending.weight)
        if self.ending.bias is not None:
            nn.init.zeros_(self.ending.bias)

    def _pad(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(value, (0, pad_width, 0, pad_height))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"NAFNet expected [B,{self.in_channels},H,W], got {tuple(value.shape)}"
            )
        original_height, original_width = value.shape[-2:]
        value = self.intro(self._pad(value))
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            value = encoder(value)
            skips.append(value)
            value = down(value)

        value = self.middle(value)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            value = up(value) + skip
            value = decoder(value)
        value = self.ending(value)
        return value[..., :original_height, :original_width]
