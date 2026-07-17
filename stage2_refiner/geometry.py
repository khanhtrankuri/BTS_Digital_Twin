"""Geometry preprocessing and overlap-blended tiled inference."""

import torch
import torch.nn.functional as F


def normalize_depth(depth, alpha, mode="robust_per_view", alpha_threshold=0.01, eps=1e-6,
                    scene_scale=None):
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = torch.nan_to_num(alpha, nan=0.0).clamp(0.0, 1.0)
    valid = torch.isfinite(depth) & (depth > 0) & (alpha > alpha_threshold)
    out = torch.zeros_like(depth)
    if not valid.any():
        return out, valid
    values = depth[valid]
    if mode == "robust_per_view":
        median = values.median()
        p05 = torch.quantile(values.float(), 0.05).to(values.dtype)
        p95 = torch.quantile(values.float(), 0.95).to(values.dtype)
        out[valid] = ((values - median) / (p95 - p05 + eps)).clamp(-1.0, 1.0)
    elif mode == "inverse_depth":
        inverse = 1.0 / values.clamp_min(eps)
        scale = inverse.median().clamp_min(eps)
        out[valid] = (inverse / scale).clamp(0.0, 4.0) / 2.0 - 1.0
    elif mode == "log_depth":
        logged = values.clamp_min(eps).log()
        out[valid] = ((logged - logged.median()) / (logged.std(unbiased=False) + eps)).clamp(-1.0, 1.0)
    elif mode == "scene_normalized":
        if scene_scale is None or scene_scale <= 0:
            raise ValueError("scene_scale must be positive for scene_normalized depth")
        out[valid] = (values / float(scene_scale)).clamp(0.0, 2.0) - 1.0
    else:
        raise ValueError(f"Unknown depth normalization: {mode}")
    return out, valid


def preprocess_geometry(depth, normal, alpha, depth_normalization="robust_per_view",
                        alpha_threshold=0.01, scene_scale=None):
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    depth, valid = normalize_depth(depth, alpha, depth_normalization, alpha_threshold, scene_scale=scene_scale)
    normal = F.normalize(torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0), dim=-3, eps=1e-6)
    normal = torch.where(valid.expand_as(normal), normal, torch.zeros_like(normal))
    return depth, normal, alpha, valid


def _blend_window(height, width, device, dtype):
    if height <= 2 or width <= 2:
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)
    wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return (wy[:, None] * wx[None, :])[None, None]


@torch.no_grad()
def tiled_inference(model, rgb, depth, normal, alpha, tile_size=512, overlap=32):
    if tile_size <= 0 or max(rgb.shape[-2:]) <= tile_size:
        return model(rgb=rgb, depth=depth, normal=normal, alpha=alpha, return_residual=True)
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile_size")
    stride = tile_size - overlap
    height, width = rgb.shape[-2:]
    output = torch.zeros_like(rgb)
    residual = torch.zeros_like(rgb)
    weights = torch.zeros_like(rgb[:, :1])
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    for y in ys:
        for x in xs:
            sl = (..., slice(y, min(y + tile_size, height)), slice(x, min(x + tile_size, width)))
            refined_tile, residual_tile = model(rgb=rgb[sl], depth=depth[sl], normal=normal[sl],
                                                alpha=alpha[sl], return_residual=True)
            window = _blend_window(refined_tile.shape[-2], refined_tile.shape[-1], rgb.device, rgb.dtype)
            output[sl] += refined_tile * window
            residual[sl] += residual_tile * window
            weights[sl] += window
    return output / weights.clamp_min(1e-6), residual / weights.clamp_min(1e-6)
