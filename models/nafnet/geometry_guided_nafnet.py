"""Residual, confidence-masked wrapper around one shared NAFNet."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .nafnet import NAFNet


class GeometryGuidedNAFNet(nn.Module):
    """Predict a bounded RGB residual and a confidence mask.

    Ground truth is intentionally absent from this interface. The first three
    input channels are always the Gaussian RGB render.
    """

    def __init__(
        self,
        in_channels: int = 10,
        width: int = 32,
        enc_blk_nums: tuple[int, ...] = (2, 2, 4, 8),
        middle_blk_num: int = 12,
        dec_blk_nums: tuple[int, ...] = (2, 2, 2, 2),
        residual_scale: float = 0.15,
        uncertainty_gating: bool = False,
        uncertainty_channel_index: int | None = 8,
        min_mask: float = 0.05,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError("GeometryGuidedNAFNet requires at least RGB input")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if uncertainty_gating and (
            uncertainty_channel_index is None
            or not 0 <= uncertainty_channel_index < in_channels
        ):
            raise ValueError("A valid uncertainty channel index is required for gating")
        self.in_channels = int(in_channels)
        self.residual_scale = float(residual_scale)
        self.uncertainty_gating = bool(uncertainty_gating)
        self.uncertainty_channel_index = uncertainty_channel_index
        self.min_mask = float(min_mask)
        self.backbone = NAFNet(
            in_channels=in_channels,
            out_channels=4,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )

    def forward(
        self,
        stage2_input: torch.Tensor,
        *,
        gaussian_rgb: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if stage2_input.ndim != 4 or stage2_input.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], got {tuple(stage2_input.shape)}"
            )
        if gaussian_rgb is None:
            gaussian_rgb = stage2_input[:, :3]
        if gaussian_rgb.shape != stage2_input[:, :3].shape:
            raise ValueError("gaussian_rgb must have shape [B,3,H,W]")

        raw = self.backbone(stage2_input)
        delta_rgb = torch.tanh(raw[:, :3])
        refine_mask = torch.sigmoid(raw[:, 3:4])
        effective_mask = refine_mask
        if self.uncertainty_gating:
            if uncertainty is None:
                assert self.uncertainty_channel_index is not None
                uncertainty = stage2_input[
                    :, self.uncertainty_channel_index : self.uncertainty_channel_index + 1
                ]
            if uncertainty.shape != refine_mask.shape:
                raise ValueError("uncertainty must have shape [B,1,H,W]")
            gate = (1.0 - uncertainty).clamp(self.min_mask, 1.0)
            effective_mask = refine_mask * gate

        correction = self.residual_scale * effective_mask * delta_rgb
        final_rgb = (gaussian_rgb + correction).clamp(0.0, 1.0)
        return {
            "final_rgb": final_rgb,
            "delta_rgb": delta_rgb,
            "refine_mask": refine_mask,
            "effective_mask": effective_mask,
            "correction": correction,
            "raw_output": raw,
        }


def build_geonaf_from_config(config: dict[str, Any]) -> GeometryGuidedNAFNet:
    """Build and validate a refiner from the YAML config dictionary."""

    model = config.get("MODEL", config)
    if str(model.get("NAME", "geonaf")).lower() != "geonaf":
        raise ValueError(f"Unsupported Stage-2 model: {model.get('NAME')!r}")
    out_channels = int(model.get("OUT_CHANNELS", 4))
    if out_channels != 4:
        raise ValueError("GeoNAF output must contain 3 delta channels and 1 mask logit")
    return GeometryGuidedNAFNet(
        in_channels=int(model.get("IN_CHANNELS", 10)),
        width=int(model.get("WIDTH", 32)),
        enc_blk_nums=tuple(int(v) for v in model.get("ENC_BLOCKS", [2, 2, 4, 8])),
        middle_blk_num=int(model.get("MIDDLE_BLOCKS", 12)),
        dec_blk_nums=tuple(int(v) for v in model.get("DEC_BLOCKS", [2, 2, 2, 2])),
        residual_scale=float(model.get("RESIDUAL_SCALE", 0.15)),
        uncertainty_gating=bool(model.get("UNCERTAINTY_GATING", False)),
        uncertainty_channel_index=model.get("UNCERTAINTY_CHANNEL_INDEX", 8),
        min_mask=float(model.get("MIN_MASK", 0.05)),
    )
