"""Native-resolution perceptual supervision helpers.

Downscaling an entire render before LPIPS is inexpensive, but it discards the
same high-frequency errors that a full-resolution submission is judged on.
These helpers keep pixels at their native sampling rate and bound memory by
selecting aligned crops.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def _validate_pair(image: torch.Tensor, target: torch.Tensor) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3,H,W]")
    if image.shape != target.shape:
        raise ValueError("image and target must have identical shapes")


def _residual_crop_origin(
    image: torch.Tensor,
    target: torch.Tensor,
    crop_height: int,
    crop_width: int,
) -> tuple[int, int]:
    """Locate a high-error crop without materializing every sliding window."""

    residual = (image.detach() - target.detach()).abs().mean(dim=0, keepdim=True)
    height, width = residual.shape[-2:]
    if crop_height == height and crop_width == width:
        return 0, 0
    stride = max(1, min(crop_height, crop_width) // 8)
    pooled = F.avg_pool2d(
        residual[None],
        kernel_size=(crop_height, crop_width),
        stride=stride,
    )[0, 0]
    flat_index = int(torch.argmax(pooled).item())
    pooled_width = pooled.shape[1]
    top = min((flat_index // pooled_width) * stride, height - crop_height)
    left = min((flat_index % pooled_width) * stride, width - crop_width)
    return int(top), int(left)


def native_resolution_crops(
    image: torch.Tensor,
    target: torch.Tensor,
    crop_size: int,
    num_crops: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned ``[N,3,Hc,Wc]`` crops at the input pixel resolution.

    The first crop targets the largest residual region. Remaining crops are
    uniformly random, balancing hard-example refinement with unbiased scene
    coverage. Python's seeded RNG makes the selection reproducible with the
    repository's existing ``safe_state`` setup.
    """

    _validate_pair(image, target)
    if int(crop_size) <= 0:
        raise ValueError("crop_size must be positive")
    if int(num_crops) <= 0:
        raise ValueError("num_crops must be positive")

    height, width = image.shape[-2:]
    crop_height = min(int(crop_size), height)
    crop_width = min(int(crop_size), width)
    origins = [
        _residual_crop_origin(image, target, crop_height, crop_width)
    ]
    for _ in range(1, int(num_crops)):
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        origins.append((top, left))

    image_crops = torch.stack(
        [
            image[:, top : top + crop_height, left : left + crop_width]
            for top, left in origins
        ]
    )
    target_crops = torch.stack(
        [
            target[:, top : top + crop_height, left : left + crop_width]
            for top, left in origins
        ]
    )
    return image_crops, target_crops
