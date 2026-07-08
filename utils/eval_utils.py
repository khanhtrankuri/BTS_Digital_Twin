import torch

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    LPIPS_AVAILABLE = False

_LPIPS_MODEL = None
_LPIPS_WARNING_PRINTED = False


def get_lpips_model():
    global _LPIPS_MODEL, _LPIPS_WARNING_PRINTED
    if not LPIPS_AVAILABLE:
        if not _LPIPS_WARNING_PRINTED:
            print("LPIPS is not installed, skip LPIPS and Score. Install it with: pip install lpips")
            _LPIPS_WARNING_PRINTED = True
        return None

    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").cuda().eval()
    return _LPIPS_MODEL


def calculate_render_metrics(image, gt_image, psnr_max, lpips_model=None):
    l1_value = l1_loss(image, gt_image).mean()
    psnr_value = psnr(image, gt_image).mean()

    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
    else:
        ssim_value = ssim(image, gt_image).mean()

    lpips_value = None
    score = None
    psnr_norm = torch.clamp(psnr_value / psnr_max, 0.0, 1.0)

    if lpips_model is not None:
        pred_lpips = image.unsqueeze(0) * 2.0 - 1.0
        gt_lpips = gt_image.unsqueeze(0) * 2.0 - 1.0
        lpips_value = lpips_model(pred_lpips, gt_lpips).mean()
        score = 0.4 * (1.0 - lpips_value) + 0.3 * ssim_value + 0.3 * psnr_norm

    return {
        "l1": l1_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "psnr_norm": psnr_norm,
        "score": score,
    }


def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        return None

    averaged = {}
    for key in metric_dicts[0]:
        values = [metrics[key] for metrics in metric_dicts if metrics[key] is not None]
        averaged[key] = torch.stack(values).mean() if values else None
    return averaged


def tensor_to_float(value):
    if value is None:
        return None
    return float(value.detach().cpu().item())


def metrics_to_floats(metrics):
    if metrics is None:
        return None
    return {key: tensor_to_float(value) for key, value in metrics.items()}


def print_metric_block(iteration, name, metrics):
    print(f"\n[ITER {iteration}] Evaluating {name}:")
    if metrics is None:
        print("  Ground-truth test images not found, skip metrics for private set.")
        return

    labels = {
        "l1": "L1",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "psnr_norm": "PSNR_norm",
        "score": "Score",
    }
    for key in ["l1", "psnr", "ssim", "lpips", "psnr_norm", "score"]:
        value = metrics.get(key)
        if value is None:
            print(f"  {labels[key]}: skipped")
        else:
            print(f"  {labels[key]}: {tensor_to_float(value):.6f}")


def write_tensorboard_metrics(tb_writer, name, iteration, metrics):
    if tb_writer is None or metrics is None:
        return

    tag_map = {
        "psnr": "psnr",
        "ssim": "ssim",
        "lpips": "lpips",
        "score": "score",
    }
    for key, suffix in tag_map.items():
        value = metrics.get(key)
        if value is not None:
            tb_writer.add_scalar(f"{name}/{suffix}", tensor_to_float(value), iteration)
