"""Window-persistent, multi-view Gaussian densification primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.densification_utils import robust_normalize_score


@dataclass
class PersistentScores:
    total: torch.Tensor
    original_gradient: torch.Tensor
    absolute_gradient: torch.Tensor
    residual: torch.Tensor
    edge: torch.Tensor
    multiview: torch.Tensor
    depth: torch.Tensor
    burstiness: torch.Tensor


def gradient_burstiness(mean_abs_gradient: torch.Tensor, mean_squared_gradient: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Measure how much gradient energy is caused by intermittent bursts."""

    mean_abs_gradient = torch.nan_to_num(mean_abs_gradient).clamp_min(0.0)
    mean_squared_gradient = torch.nan_to_num(mean_squared_gradient).clamp_min(0.0)
    variance_like = (mean_squared_gradient - mean_abs_gradient.square()).clamp_min(0.0)
    return (variance_like / (mean_squared_gradient + eps)).clamp(0.0, 1.0)


def view_direction_bins(directions: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Quantize unit directions into a compact latitude/longitude histogram."""

    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape [N,3]")
    if num_bins < 1 or num_bins > 62:
        raise ValueError("num_bins must be in [1,62] for the int64 support bitset")
    directions = torch.nn.functional.normalize(directions, dim=-1, eps=1e-6)
    elevation_bins = max(1, int(round(num_bins ** 0.5)))
    azimuth_bins = max(1, (num_bins + elevation_bins - 1) // elevation_bins)
    azimuth = torch.atan2(directions[:, 1], directions[:, 0])
    elevation = torch.asin(directions[:, 2].clamp(-1.0, 1.0))
    azimuth_index = ((azimuth + torch.pi) / (2.0 * torch.pi) * azimuth_bins).long().clamp(0, azimuth_bins - 1)
    elevation_index = ((elevation + 0.5 * torch.pi) / torch.pi * elevation_bins).long().clamp(0, elevation_bins - 1)
    return (elevation_index * azimuth_bins + azimuth_index).clamp_max(num_bins - 1)


def update_view_support(bitset: torch.Tensor, gaussian_directions: torch.Tensor,
                        visible_mask: torch.Tensor, num_bins: int) -> None:
    """OR the current view-direction bin into each visible Gaussian bitset."""

    visible = visible_mask.reshape(-1).bool()
    if bitset.shape[0] != visible.shape[0] or bitset.dtype != torch.int64:
        raise ValueError("bitset must be int64 with one row per Gaussian")
    if not visible.any():
        return
    bins = view_direction_bins(gaussian_directions[visible], num_bins)
    values = torch.ones_like(bins, dtype=torch.int64) << bins
    bitset[visible, 0] = torch.bitwise_or(bitset[visible, 0], values)


def bitset_popcount(bitset: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Count occupied view bins without allocating an N-by-N tensor."""

    values = bitset.reshape(-1).to(torch.int64)
    result = torch.zeros_like(values)
    for index in range(int(num_bins)):
        result += torch.bitwise_and(torch.bitwise_right_shift(values, index), 1)
    return result


def compute_persistent_scores(*, original_grad: torch.Tensor, abs_grad: torch.Tensor,
                              grad_sq: torch.Tensor, residual: torch.Tensor,
                              edge: torch.Tensor, unique_views: torch.Tensor,
                              depth_support: torch.Tensor, sky_support: torch.Tensor,
                              low_parallax_support: torch.Tensor, valid_mask: torch.Tensor,
                              cfg) -> PersistentScores:
    """Compute robust, fully configurable persistent densification scores."""

    burst = gradient_burstiness(abs_grad, grad_sq)
    signals = {
        "original": robust_normalize_score(original_grad, valid_mask),
        "absolute": robust_normalize_score(abs_grad, valid_mask),
        "residual": robust_normalize_score(residual, valid_mask),
        "edge": robust_normalize_score(edge, valid_mask),
        "multiview": robust_normalize_score(unique_views.float(), valid_mask),
        "depth": robust_normalize_score(depth_support, valid_mask),
    }
    total = (
        float(cfg.densification_original_grad_weight) * signals["original"]
        + float(cfg.densification_abs_grad_weight) * signals["absolute"]
        + float(cfg.densification_residual_weight) * signals["residual"]
        + float(cfg.densification_edge_weight) * signals["edge"]
        + float(cfg.densification_multiview_weight) * signals["multiview"]
        + float(cfg.densification_depth_support_weight) * signals["depth"]
    )
    if getattr(cfg, "densification_burst_suppression_enabled", True):
        total = total - float(cfg.densification_burst_penalty_weight) * burst
    total = total.clamp_min(0.0)
    total = total * torch.where(
        sky_support > 0.5, total.new_tensor(float(cfg.densification_sky_score_multiplier)), 1.0)
    total = total * torch.where(
        low_parallax_support > 0.5,
        total.new_tensor(float(cfg.densification_low_parallax_score_multiplier)), 1.0)
    return PersistentScores(total, signals["original"], signals["absolute"], signals["residual"],
                            signals["edge"], signals["multiview"], signals["depth"], burst)


def update_persistent_window(score_ema: torch.Tensor, hit_ema: torch.Tensor,
                             hit_count: torch.Tensor, recent_mask: torch.Tensor,
                             window_hit: torch.Tensor, score: torch.Tensor,
                             decay: float, recent_window_count: int) -> None:
    """Update persistent state in-place after one densification window."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("persistence decay must be in [0,1)")
    if not 1 <= recent_window_count <= 62:
        raise ValueError("recent_window_count must be in [1,62]")
    hit = window_hit.reshape(-1, 1).to(score_ema.dtype)
    score_ema.mul_(decay).add_(score.reshape(-1, 1), alpha=1.0 - decay)
    hit_ema.mul_(decay).add_(hit, alpha=1.0 - decay)
    hit_count.add_(hit.to(hit_count.dtype))
    keep_bits = (1 << recent_window_count) - 1
    recent_mask.copy_(torch.bitwise_and(torch.bitwise_left_shift(recent_mask, 1) | hit.to(torch.int64), keep_bits))


def recent_hit_count(recent_mask: torch.Tensor, recent_window_count: int) -> torch.Tensor:
    return bitset_popcount(recent_mask, recent_window_count)
