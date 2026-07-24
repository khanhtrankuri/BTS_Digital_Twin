"""Losses and image-quality metrics for the residual GeoNAF stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from utils.loss_utils import ssim


def charbonnier_loss(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + float(eps) ** 2).mean()


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sobel x/y gradients for every RGB channel."""

    if image.ndim != 4:
        raise ValueError("image_gradients expects [B,C,H,W]")
    channels = image.shape[1]
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    kernel_x = kernel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    gradient_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
    gradient_y = F.conv2d(
        image, kernel_x.transpose(-1, -2), padding=1, groups=channels
    )
    return gradient_x, gradient_y


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y = image_gradients(prediction)
    target_x, target_y = image_gradients(target)
    return 0.5 * (
        F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
    )


def identity_preservation_loss(
    final_rgb: torch.Tensor,
    gaussian_rgb: torch.Tensor,
    target: torch.Tensor,
    k: float = 10.0,
) -> torch.Tensor:
    low_error_mask = torch.exp(
        -float(k) * (gaussian_rgb - target).abs().mean(dim=1, keepdim=True)
    ).detach()
    return (low_error_mask * (final_rgb - gaussian_rgb).abs()).mean()


class Stage2Loss(nn.Module):
    """Weighted Stage-2 objective with optional externally provided LPIPS."""

    def __init__(
        self,
        loss_config: dict[str, Any],
        *,
        perceptual_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = {
            "charbonnier": float(loss_config.get("CHARBONNIER", 1.0)),
            "ssim": float(loss_config.get("SSIM", 0.2)),
            "edge": float(loss_config.get("EDGE", 0.1)),
            "perceptual": float(loss_config.get("PERCEPTUAL", 0.0)),
            "residual": float(loss_config.get("RESIDUAL", 0.01)),
            "mask": float(loss_config.get("MASK", 0.005)),
            "identity": float(loss_config.get("IDENTITY", 0.05)),
        }
        self.charbonnier_eps = float(loss_config.get("CHARBONNIER_EPS", 1e-3))
        self.identity_k = float(loss_config.get("IDENTITY_K", 10.0))
        self.perceptual_model = perceptual_model
        if self.weights["perceptual"] > 0.0 and perceptual_model is None:
            raise RuntimeError(
                "LOSS.PERCEPTUAL is positive but no local perceptual model was loaded"
            )

    def forward(
        self,
        output: dict[str, torch.Tensor],
        gaussian_rgb: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        final_rgb = output["final_rgb"]
        char = charbonnier_loss(final_rgb, target, self.charbonnier_eps)
        ssim_term = 1.0 - ssim(final_rgb, target)
        edge_term = edge_loss(final_rgb, target)
        residual_term = output["delta_rgb"].abs().mean()
        mask_term = output["refine_mask"].mean()
        identity_term = identity_preservation_loss(
            final_rgb, gaussian_rgb, target, self.identity_k
        )
        perceptual = final_rgb.new_zeros(())
        if self.perceptual_model is not None and self.weights["perceptual"] > 0.0:
            perceptual = self.perceptual_model(
                final_rgb * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()
        terms = {
            "charbonnier": char,
            "ssim": ssim_term,
            "edge": edge_term,
            "perceptual": perceptual,
            "residual": residual_term,
            "mask": mask_term,
            "identity": identity_term,
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        return {"total": total, **terms}


def load_lpips_if_available(
    device: torch.device,
    *,
    allow_weight_download: bool = False,
) -> nn.Module | None:
    """Load LPIPS only when its ImageNet backbone is already local.

    ``allow_weight_download`` is deliberately explicit so training never
    silently accesses the internet.
    """

    try:
        import lpips
    except ImportError:
        return None

    alexnet_cache = (
        Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    )
    if not alexnet_cache.is_file() and not allow_weight_download:
        return None
    model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def stage2_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_model: nn.Module | None = None,
) -> dict[str, float]:
    """Compute PSNR, SSIM, optional LPIPS and gradient error."""

    mse = F.mse_loss(prediction, target).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    ssim_value = ssim(prediction, target)
    gradient = edge_loss(prediction, target)
    edge_psnr = -10.0 * torch.log10(gradient.square().clamp_min(1e-12))
    lpips_value = None
    if lpips_model is not None:
        lpips_value = float(
            lpips_model(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
            .mean()
            .item()
        )
    return {
        "psnr": float(psnr.item()),
        "ssim": float(ssim_value.item()),
        "lpips": lpips_value,
        "gradient_error": float(gradient.item()),
        "edge_psnr": float(edge_psnr.item()),
    }
