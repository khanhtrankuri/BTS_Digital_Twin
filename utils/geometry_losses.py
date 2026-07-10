"""Numerically robust, independently testable losses for BTS-GeoGS.

All functions accept both ``[C, H, W]`` and batched tensors.  Optional masks
and confidence tensors are broadcast against the prediction tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_mask(pred, target, confidence=None, valid_mask=None):
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if valid_mask is not None:
        mask = mask & valid_mask.bool()
    if confidence is not None:
        mask = mask & torch.isfinite(confidence) & (confidence > 0)
    return mask


def _zero_with_grad(tensor):
    return tensor.sum() * 0.0


def scale_shift_invariant_depth_loss(pred_depth, target_depth, confidence=None,
                                     valid_mask=None, eps=1e-6):
    """Robust L1 after least-squares scale/shift alignment of ``pred_depth``.

    The alignment is solved independently for every batch item, so it remains
    well-defined for the common ``[1, H, W]`` camera convention.
    """
    pred = torch.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target_depth, nan=0.0, posinf=0.0, neginf=0.0)
    mask = _as_mask(pred_depth, target_depth, confidence, valid_mask)
    mask = mask & (pred > eps) & (target > eps)
    weights = torch.ones_like(pred) if confidence is None else torch.clamp(
        torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)

    # A single camera is normally unbatched. Flattening preserves correctness
    # and avoids unstable tiny per-row linear systems.
    p, t, w = pred[mask], target[mask], weights[mask]
    if p.numel() < 2 or torch.sum(w) <= eps:
        return _zero_with_grad(pred_depth)
    w = w / (w.sum() + eps)
    p_mean, t_mean = (w * p).sum(), (w * t).sum()
    var_p = (w * (p - p_mean).square()).sum()
    scale = (w * (p - p_mean) * (t - t_mean)).sum() / (var_p + eps)
    shift = t_mean - scale * p_mean
    return (w * (scale * p + shift - t).abs()).sum()


def normal_consistency_loss(pred_normal, target_normal, confidence=None,
                            valid_mask=None, use_abs_cosine=True, eps=1e-6):
    """Cosine normal loss, optionally invariant to normal orientation."""
    pred = F.normalize(torch.nan_to_num(pred_normal), dim=-3, eps=eps)
    target = F.normalize(torch.nan_to_num(target_normal), dim=-3, eps=eps)
    cosine = (pred * target).sum(dim=-3, keepdim=True)
    if use_abs_cosine:
        cosine = cosine.abs()
    # Normal validity is evaluated per pixel rather than per channel.
    mask = torch.isfinite(pred_normal).all(dim=-3, keepdim=True)
    mask &= torch.isfinite(target_normal).all(dim=-3, keepdim=True)
    if valid_mask is not None:
        mask &= valid_mask.bool()
    weights = torch.ones_like(cosine) if confidence is None else torch.clamp(
        torch.nan_to_num(confidence), min=0.0)
    weights = weights * mask
    if torch.sum(weights) <= eps:
        return _zero_with_grad(pred_normal)
    return (weights * (1.0 - cosine)).sum() / (weights.sum() + eps)


def edge_weighted_l1_loss(pred_rgb, target_rgb, edge_map, gamma=2.0, eps=1e-6):
    """L1 RGB loss that up-weights Sobel edges without replacing base RGB loss."""
    edge = torch.clamp(torch.nan_to_num(edge_map), min=0.0)
    weights = 1.0 + float(gamma) * edge
    error = torch.nan_to_num((pred_rgb - target_rgb).abs()).mean(dim=-3, keepdim=True)
    return (weights * error).sum() / (weights.sum() + eps)


def gaussian_scale_regularization(scales, max_scale, max_anisotropy_ratio=None,
                                  eps=1e-6):
    """Penalize only over-large or overly anisotropic Gaussian scales."""
    if scales.numel() == 0:
        return _zero_with_grad(scales)
    scales = torch.clamp(torch.nan_to_num(scales), min=eps)
    loss = F.relu(scales.max(dim=-1).values - float(max_scale)).square().mean()
    if max_anisotropy_ratio is not None and max_anisotropy_ratio > 0:
        ratio = scales.max(dim=-1).values / (scales.min(dim=-1).values + eps)
        loss = loss + F.relu(ratio - float(max_anisotropy_ratio)).square().mean()
    return loss


def sobel_edge_map(rgb, eps=1e-6):
    """Return a normalized, cached-friendly Sobel edge map using only PyTorch."""
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    gray = 0.2989 * rgb[:, :1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]
    kx = rgb.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + eps)
    edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + eps)
    return edge.squeeze(0)


def get_loss_weights(iteration, cfg):
    """Three-stage loss schedule driven entirely by optimization parameters."""
    if not getattr(cfg, "geometry_aware", False):
        return {name: 0.0 for name in ("depth", "normal", "edge", "scale")}
    warmup = max(0, int(cfg.geometry_warmup_until))
    densify_until = max(warmup + 1, int(cfg.densify_until_iter))
    if iteration <= warmup:
        geometry_decay, edge_factor = 1.0, float(cfg.geometry_edge_warmup_factor)
    elif iteration <= densify_until:
        geometry_decay = 1.0 - (iteration - warmup) / float(densify_until - warmup)
        edge_factor = 1.0
    else:
        geometry_decay, edge_factor = 0.0, 1.0
    return {
        "depth": float(cfg.depth_loss_weight) * geometry_decay if cfg.depth_loss_enabled else 0.0,
        "normal": float(cfg.normal_loss_weight) * geometry_decay if cfg.normal_loss_enabled else 0.0,
        "edge": float(cfg.edge_loss_weight) * edge_factor if cfg.edge_loss_enabled else 0.0,
        "scale": float(cfg.scale_reg_weight) if cfg.scale_reg_enabled else 0.0,
    }
