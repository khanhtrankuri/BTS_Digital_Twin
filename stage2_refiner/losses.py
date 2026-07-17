"""Metric-oriented, geometry-aware losses for the residual refiner."""

import torch
from torch import nn
import torch.nn.functional as F


def _ssim(pred, target, window_size=11):
    channels = pred.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    var_x = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_x.square()
    var_y = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y.square()
    covariance = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2) + 1e-8)
    return score.mean()


def sobel_gradients(image):
    gray = image.mean(1, keepdim=True)
    kx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    return F.conv2d(gray, kx, padding=1), F.conv2d(gray, kx.transpose(-1, -2), padding=1)


def get_stage2_loss_weights(current_step, total_steps, cfg):
    progress = current_step / max(1, total_steps)
    schedules = cfg.get("LOSS_SCHEDULE", {})
    stage = "A" if progress < 0.2 else ("B" if progress < 0.8 else "C")
    loss = cfg.get("LOSS", {})
    weights = {key.lower().replace("_weight", ""): float(value)
               for key, value in loss.items() if key.endswith("_WEIGHT")}
    if stage in schedules:
        weights.update({key.lower().replace("_weight", ""): float(value)
                        for key, value in schedules[stage].items()})
    return weights


class Stage2Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        loss_cfg = cfg.get("LOSS", {})
        self.depth_edge_lambda = float(loss_cfg.get("DEPTH_EDGE_LAMBDA", 0.25))
        self.normal_edge_lambda = float(loss_cfg.get("NORMAL_EDGE_LAMBDA", 0.10))

    def forward(self, refined, residual, target, rgb_input, depth, normal, alpha,
                current_step=0, total_steps=1, gate=None):
        weights = get_stage2_loss_weights(current_step, total_steps, self.cfg)
        l1 = F.l1_loss(refined, target)
        mse = F.mse_loss(refined, target)
        dssim = 1.0 - _ssim(refined, target)
        pred_gx, pred_gy = sobel_gradients(refined)
        target_gx, target_gy = sobel_gradients(target)
        depth_gx, depth_gy = sobel_gradients(depth.repeat(1, 3, 1, 1))
        normal_gx, normal_gy = sobel_gradients((normal + 1.0) * 0.5)
        geo_weight = 1.0 + self.depth_edge_lambda * (depth_gx.abs() + depth_gy.abs())
        geo_weight += self.normal_edge_lambda * (normal_gx.abs() + normal_gy.abs())
        edge = (geo_weight * ((pred_gx - target_gx).abs() + (pred_gy - target_gy).abs())).mean()
        residual_reg = residual.abs().mean()
        low_pred = F.avg_pool2d(refined, 7, stride=1, padding=3)
        low_target = F.avg_pool2d(target, 7, stride=1, padding=3)
        low_frequency = F.l1_loss(low_pred, low_target)
        gate_reg = gate.abs().mean() if gate is not None else refined.new_zeros(())
        terms = {"l1": l1, "mse": mse, "dssim": dssim, "edge": edge,
                 "residual": residual_reg, "low_frequency": low_frequency, "gate": gate_reg}
        total = sum(weights.get(name, 0.0) * value for name, value in terms.items())
        terms["total"] = total
        terms["stage"] = "A" if current_step / max(1, total_steps) < .2 else ("B" if current_step / max(1, total_steps) < .8 else "C")
        return terms


def ssim_metric(pred, target):
    return _ssim(pred, target)
