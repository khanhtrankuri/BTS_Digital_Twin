"""Memory-bounded tiled inference for the GeoNAF refiner."""

from __future__ import annotations

import torch


_BLENDED_KEYS = (
    "final_rgb",
    "delta_rgb",
    "refine_mask",
    "effective_mask",
    "correction",
)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_window(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # A small positive floor prevents zero denominators at outer image borders.
    vertical = torch.hann_window(
        max(height, 2), periodic=False, device=device, dtype=dtype
    )[:height].clamp_min(1e-3)
    horizontal = torch.hann_window(
        max(width, 2), periodic=False, device=device, dtype=dtype
    )[:width].clamp_min(1e-3)
    return (vertical[:, None] * horizontal[None, :])[None, None]


def tiled_refine(
    model,
    stage2_input: torch.Tensor,
    *,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, torch.Tensor]:
    """Run full-image or overlap-blended tiled residual refinement."""

    if stage2_input.ndim != 4:
        raise ValueError("tiled_refine expects [B,C,H,W]")
    if stage2_input.shape[0] != 1:
        if tile_size > 0:
            raise ValueError("Tiled inference currently requires batch size 1")
        return model(stage2_input)
    height, width = stage2_input.shape[-2:]
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return model(stage2_input)
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile_size")

    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    accumulators: dict[str, torch.Tensor] = {}
    denominator = stage2_input.new_zeros((1, 1, height, width))

    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = stage2_input[..., top:bottom, left:right]
            tile_output = model(tile)
            window = _blend_window(
                bottom - top,
                right - left,
                device=tile.device,
                dtype=tile.dtype,
            )
            denominator[..., top:bottom, left:right] += window
            for key in _BLENDED_KEYS:
                value = tile_output[key]
                if key not in accumulators:
                    accumulators[key] = value.new_zeros(
                        (1, value.shape[1], height, width)
                    )
                accumulators[key][..., top:bottom, left:right] += value * window

    denominator = denominator.clamp_min(1e-8)
    result = {key: value / denominator for key, value in accumulators.items()}
    # Preserve the core residual invariant after floating-point blending.
    result["final_rgb"] = result["final_rgb"].clamp(0.0, 1.0)
    return result
