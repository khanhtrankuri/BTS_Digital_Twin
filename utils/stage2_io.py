"""Shared configuration, checkpoint, manifest and image I/O for Stage 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from models.nafnet import build_geonaf_from_config
from utils.config_utils import load_yaml_with_base


def load_stage2_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config = load_yaml_with_base(path)
    required = ("MODEL", "DATA", "NORMALIZATION", "LOSS", "TRAIN", "INFERENCE")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Stage-2 config is missing sections: {missing}")
    return config


def save_tensor_image(value: torch.Tensor, path: str | os.PathLike[str]) -> None:
    """Save one RGB or grayscale tensor without RGB/BGR conversion."""

    value = value.detach().float().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError(f"Expected [1,H,W] or [3,H,W], got {tuple(value.shape)}")
    array = (
        value.clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).numpy()
    )
    if array.shape[2] == 1:
        array = array[..., 0]
        image = Image.fromarray(array, mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_rgb_image(path: str | os.PathLike[str]) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def discover_manifests(root: str | os.PathLike[str]) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    direct = root / "manifest.json"
    if direct.is_file():
        return [direct]
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.json found under {root}")
    return manifests


def load_manifest_entries(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for manifest_path in discover_manifests(root):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = payload.get("frames", payload) if isinstance(payload, dict) else payload
        if not isinstance(current, list):
            raise ValueError(f"Manifest frames must be a list: {manifest_path}")
        for frame in current:
            item = dict(frame)
            item["_manifest_dir"] = str(manifest_path.parent.resolve())
            entries.append(item)
    if not entries:
        raise ValueError(f"No frames found in manifests under {root}")
    return entries


def resolve_entry_path(entry: dict[str, Any], key: str) -> Path:
    if key not in entry:
        raise KeyError(f"Manifest entry is missing {key!r}")
    value = Path(entry[key])
    if not value.is_absolute():
        value = Path(entry["_manifest_dir"]) / value
    return value


def save_stage2_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    model_config = config["MODEL"]
    normalization = config["NORMALIZATION"]
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metrics": dict(best_metrics),
        "config": config,
        "expected_input_channels": int(model_config["IN_CHANNELS"]),
        "residual_scale": float(model_config["RESIDUAL_SCALE"]),
        "normalization": dict(normalization),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_refiner_checkpoint(
    path: str | os.PathLike[str],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid Stage-2 checkpoint: {path}")
    expected = int(config["MODEL"]["IN_CHANNELS"])
    stored = int(payload.get("expected_input_channels", expected))
    if stored != expected:
        raise ValueError(
            f"Checkpoint expects {stored} input channels but config declares {expected}"
        )
    configured_scale = float(config["MODEL"]["RESIDUAL_SCALE"])
    stored_scale = float(payload.get("residual_scale", configured_scale))
    if abs(stored_scale - configured_scale) > 1e-9:
        raise ValueError(
            f"Checkpoint residual scale {stored_scale} does not match config "
            f"{configured_scale}"
        )
    stored_normalization = payload.get("normalization")
    if isinstance(stored_normalization, dict):
        for key in (
            "ALPHA_THRESHOLD",
            "DEPTH_CLIP",
            "VARIANCE_CLIP",
            "EPS",
            "MIN_VALID_PIXELS",
        ):
            if key not in stored_normalization or key not in config["NORMALIZATION"]:
                continue
            stored_value = float(stored_normalization[key])
            configured_value = float(config["NORMALIZATION"][key])
            if abs(stored_value - configured_value) > 1e-12:
                raise ValueError(
                    f"Checkpoint normalization {key}={stored_value} does not "
                    f"match config value {configured_value}"
                )
    model = build_geonaf_from_config(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload
