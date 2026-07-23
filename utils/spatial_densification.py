"""Spatially balanced per-tile densification budget helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SpatialSelection:
    selected: torch.Tensor
    tile_energy: torch.Tensor
    tile_budget: torch.Tensor
    tile_selected: torch.Tensor
    top_decile_budget_fraction: float


def tile_error_energy(residual_map: torch.Tensor, edge_map: torch.Tensor,
                      tile_size: int = 64, edge_weight: float = 0.25) -> torch.Tensor:
    """Return mean residual plus weighted mean edge energy for each tile."""

    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    residual = residual_map[0] if residual_map.ndim == 3 and residual_map.shape[0] == 1 else residual_map
    edge = edge_map[0] if edge_map.ndim == 3 and edge_map.shape[0] == 1 else edge_map
    if residual.ndim != 2 or residual.shape != edge.shape:
        raise ValueError("residual_map and edge_map must share [H,W] or [1,H,W] shape")
    height, width = residual.shape
    pad_h = (-height) % tile_size
    pad_w = (-width) % tile_size
    values = torch.stack((torch.nan_to_num(residual), torch.nan_to_num(edge)), dim=0)[None]
    valid = torch.ones((1, 1, height, width), device=residual.device, dtype=residual.dtype)
    values = F.pad(values, (0, pad_w, 0, pad_h))
    valid = F.pad(valid, (0, pad_w, 0, pad_h))
    sums = F.avg_pool2d(values, tile_size, stride=tile_size, divisor_override=1)[0]
    counts = F.avg_pool2d(valid, tile_size, stride=tile_size, divisor_override=1)[0, 0].clamp_min(1.0)
    means = sums / counts[None]
    return (means[0] + float(edge_weight) * means[1]).clamp_min(0.0)


def allocate_tile_budget(tile_energy: torch.Tensor, total_budget: int, gamma: float = 0.5,
                         minimum: int = 4, maximum: int = 256) -> torch.Tensor:
    """Allocate an integer bounded budget without exceeding ``total_budget``."""

    if total_budget < 0 or minimum < 0 or maximum < 1 or minimum > maximum:
        raise ValueError("invalid tile budget bounds")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    energy = torch.nan_to_num(tile_energy, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    flat = energy.reshape(-1)
    budget = torch.zeros_like(flat, dtype=torch.long)
    active = flat > 0
    active_count = int(active.sum().item())
    if total_budget == 0 or active_count == 0:
        return budget.reshape_as(energy)
    capacity = min(int(total_budget), active_count * int(maximum))
    active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
    if capacity < active_count * int(minimum):
        ranked = active_indices[torch.argsort(flat[active_indices], descending=True)]
        budget[ranked[:capacity]] = 1
        return budget.reshape_as(energy)
    budget[active] = int(minimum)
    remaining = capacity - active_count * int(minimum)
    while remaining > 0:
        available = active & (budget < int(maximum))
        if not available.any():
            break
        indices = torch.nonzero(available, as_tuple=False).squeeze(1)
        weights = flat[indices].pow(float(gamma))
        shares = remaining * weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        additions = torch.minimum(torch.floor(shares).long(), int(maximum) - budget[indices])
        if int(additions.sum().item()) == 0:
            order = torch.argsort(shares, descending=True)
            take = min(remaining, int(order.numel()))
            additions[order[:take]] = 1
        budget[indices] += additions
        remaining -= int(additions.sum().item())
    return budget.reshape_as(energy)


def spatially_balanced_topk(candidate_mask: torch.Tensor, score: torch.Tensor,
                            centers: torch.Tensor, tile_energy: torch.Tensor,
                            image_width: int, image_height: int, total_budget: int,
                            tile_size: int, gamma: float, minimum: int,
                            maximum: int) -> SpatialSelection:
    """Select each Gaussian at most once from the tile containing its center.

    A Gaussian center lies inside its own ellipse footprint, so this assignment
    is a conservative footprint-overlap test without an ``N x tiles`` tensor.
    """

    if candidate_mask.shape != score.shape or centers.shape != (score.numel(), 2):
        raise ValueError("candidate_mask/score must be [N] and centers [N,2]")
    budget = allocate_tile_budget(tile_energy, total_budget, gamma, minimum, maximum)
    tiles_y, tiles_x = tile_energy.shape
    pixel_x = (centers[:, 0] + 1.0) * 0.5 * float(image_width)
    pixel_y = (centers[:, 1] + 1.0) * 0.5 * float(image_height)
    tile_x = torch.floor(pixel_x / int(tile_size)).long().clamp(0, tiles_x - 1)
    tile_y = torch.floor(pixel_y / int(tile_size)).long().clamp(0, tiles_y - 1)
    inside = (centers[:, 0].abs() <= 1.0) & (centers[:, 1].abs() <= 1.0)
    tile_id = tile_y * tiles_x + tile_x
    selected = torch.zeros_like(candidate_mask)
    selected_per_tile = torch.zeros_like(budget.reshape(-1))
    for current_tile in torch.nonzero(budget.reshape(-1) > 0, as_tuple=False).squeeze(1).tolist():
        members = candidate_mask & inside & (tile_id == int(current_tile))
        indices = torch.nonzero(members, as_tuple=False).squeeze(1)
        count = min(int(budget.reshape(-1)[current_tile].item()), int(indices.numel()))
        if count:
            chosen = indices[torch.topk(score[indices], k=count).indices]
            selected[chosen] = True
            selected_per_tile[current_tile] = count
    flat_energy = tile_energy.reshape(-1)
    top_count = max(1, int(round(0.1 * flat_energy.numel())))
    top_tiles = torch.topk(flat_energy, k=top_count).indices
    fraction = float(budget.reshape(-1)[top_tiles].sum().item() / max(1, budget.sum().item()))
    return SpatialSelection(selected, tile_energy, budget,
                            selected_per_tile.reshape_as(budget), fraction)

