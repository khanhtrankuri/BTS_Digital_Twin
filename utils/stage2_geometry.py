"""Geometry sanitization and per-image robust normalization for BTS GeoNAF-GS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


DEFAULT_COMPONENTS = (
    "gaussian_rgb",
    "normalized_depth",
    "normal",
    "alpha",
    "uncertainty",
    "normalized_variance",
)

COMPONENT_CHANNELS = {
    "gaussian_rgb": 3,
    "normalized_depth": 1,
    "normal": 3,
    "alpha": 1,
    "uncertainty": 1,
    "normalized_variance": 1,
    "foreground_mask": 1,
    "thin_structure_mask": 1,
    "dynamic_mask": 1,
    "sky_background_mask": 1,
}


def _as_batched(value: torch.Tensor, channels: int, name: str) -> tuple[torch.Tensor, bool]:
    if value.ndim == 3:
        value = value.unsqueeze(0)
        squeeze = True
    elif value.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"{name} must have shape [C,H,W] or [B,C,H,W]")
    if value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channel(s), got {value.shape[1]}")
    return value.float(), squeeze


def robust_normalize_geometry(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    depth_variance: torch.Tensor,
    *,
    alpha_threshold: float = 0.01,
    depth_clip: float = 5.0,
    variance_clip: float = 5.0,
    eps: float = 1e-6,
    min_valid_pixels: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize metric depth and variance independently for every image.

    All statistics are evaluated in float32. Invalid or underspecified images
    safely return zeros, and every output is finite.
    """

    if depth_clip <= 0.0 or variance_clip <= 0.0 or eps <= 0.0:
        raise ValueError("Normalization clips and eps must be positive")
    depth, squeeze_depth = _as_batched(depth, 1, "depth")
    alpha, squeeze_alpha = _as_batched(alpha, 1, "alpha")
    variance, squeeze_variance = _as_batched(
        depth_variance, 1, "depth_variance"
    )
    if squeeze_depth != squeeze_alpha or squeeze_depth != squeeze_variance:
        raise ValueError("depth, alpha and variance must have matching ranks")
    if depth.shape != alpha.shape or depth.shape != variance.shape:
        raise ValueError("depth, alpha and variance must have identical shapes")

    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    variance = torch.nan_to_num(
        variance, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    normalized_depth = torch.zeros_like(depth)
    normalized_variance = torch.zeros_like(variance)

    for index in range(depth.shape[0]):
        valid = (alpha[index] > float(alpha_threshold)) & (depth[index] > 0.0)
        valid &= torch.isfinite(depth[index])
        if int(valid.sum().item()) < int(min_valid_pixels):
            continue
        values = depth[index][valid]
        median = values.median()
        mad = (values - median).abs().median()
        current_depth = ((depth[index] - median) / (mad + eps)).clamp(
            -depth_clip, depth_clip
        )
        current_depth = current_depth / depth_clip
        current_depth = torch.where(valid, current_depth, torch.zeros_like(current_depth))

        relative_variance = variance[index] / (depth[index].square() + eps)
        current_variance = torch.log1p(relative_variance).clamp(0.0, variance_clip)
        current_variance = current_variance / variance_clip
        current_variance = torch.where(
            valid, current_variance, torch.zeros_like(current_variance)
        )
        normalized_depth[index] = torch.nan_to_num(current_depth)
        normalized_variance[index] = torch.nan_to_num(current_variance)

    if squeeze_depth:
        return normalized_depth[0], normalized_variance[0]
    return normalized_depth, normalized_variance


def expected_input_channels(components: Sequence[str]) -> int:
    unknown = [name for name in components if name not in COMPONENT_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown Stage-2 input component(s): {unknown}")
    return sum(COMPONENT_CHANNELS[name] for name in components)


def assemble_stage2_input(
    maps: Mapping[str, torch.Tensor],
    components: Sequence[str] = DEFAULT_COMPONENTS,
    *,
    expected_channels: int | None = None,
) -> torch.Tensor:
    """Concatenate configured maps after strict shape/finite validation."""

    values: list[torch.Tensor] = []
    spatial_shape: tuple[int, ...] | None = None
    rank: int | None = None
    for name in components:
        if name not in maps:
            raise KeyError(f"Stage-2 input is missing component {name!r}")
        value = maps[name].float()
        channels = COMPONENT_CHANNELS.get(name)
        if value.ndim not in (3, 4) or value.shape[-3] != channels:
            raise ValueError(
                f"{name} must be [C,H,W] or [B,C,H,W] with C={channels}, "
                f"got {tuple(value.shape)}"
            )
        if rank is None:
            rank = value.ndim
            spatial_shape = tuple(value.shape[:-3] + value.shape[-2:])
        elif value.ndim != rank or tuple(value.shape[:-3] + value.shape[-2:]) != spatial_shape:
            raise ValueError("All Stage-2 maps must share batch and spatial dimensions")
        values.append(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    result = torch.cat(values, dim=-3)
    wanted = expected_input_channels(components)
    if expected_channels is not None and int(expected_channels) != wanted:
        raise ValueError(
            f"Config declares {expected_channels} input channels but components provide {wanted}"
        )
    if result.shape[-3] != wanted:
        raise AssertionError("Internal Stage-2 channel assembly error")
    return result


def prepare_stage2_input_from_render(
    render_package: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    *,
    optional_masks: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize a renderer package and assemble one batched model input."""

    model_config = config["MODEL"]  # type: ignore[index]
    data_config = config["DATA"]  # type: ignore[index]
    norm_config = config["NORMALIZATION"]  # type: ignore[index]
    for name in (
        "render",
        "depth",
        "normal",
        "alpha",
        "uncertainty",
        "depth_variance",
    ):
        if name not in render_package or render_package[name] is None:
            raise KeyError(
                f"Gaussian renderer package is missing required map {name!r}; "
                "render with render_geometry=True"
            )
    normalized_depth, normalized_variance = robust_normalize_geometry(
        render_package["depth"],
        render_package["alpha"],
        render_package["depth_variance"],
        alpha_threshold=float(norm_config.get("ALPHA_THRESHOLD", 0.01)),  # type: ignore[union-attr]
        depth_clip=float(norm_config.get("DEPTH_CLIP", 5.0)),  # type: ignore[union-attr]
        variance_clip=float(norm_config.get("VARIANCE_CLIP", 5.0)),  # type: ignore[union-attr]
        eps=float(norm_config.get("EPS", 1e-6)),  # type: ignore[union-attr]
        min_valid_pixels=int(norm_config.get("MIN_VALID_PIXELS", 1)),  # type: ignore[union-attr]
    )
    maps = {
        "gaussian_rgb": render_package["render"].float().clamp(0.0, 1.0),
        "normalized_depth": normalized_depth,
        "normal": render_package["normal"].float().clamp(-1.0, 1.0),
        "alpha": render_package["alpha"].float().clamp(0.0, 1.0),
        "uncertainty": render_package["uncertainty"].float().clamp(0.0, 1.0),
        "normalized_variance": normalized_variance,
    }
    maps.update(optional_masks or {})
    components = tuple(data_config.get("INPUT_COMPONENTS", DEFAULT_COMPONENTS))  # type: ignore[union-attr]
    if bool(data_config.get("USE_SEGMENTATION", False)):  # type: ignore[union-attr]
        components += tuple(
            name
            for name in data_config.get("SEGMENTATION_MASKS", ())  # type: ignore[union-attr]
            if name not in components
        )
    stage2_input = assemble_stage2_input(
        maps,
        components,
        expected_channels=int(model_config["IN_CHANNELS"]),  # type: ignore[index]
    )
    if stage2_input.ndim == 3:
        stage2_input = stage2_input.unsqueeze(0)
    return stage2_input, maps
