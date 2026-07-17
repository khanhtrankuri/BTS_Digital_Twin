"""Shared config, reproducibility and image helpers."""

from copy import deepcopy
import json
import os
import random

import numpy as np
import torch
from PIL import Image


def deep_update(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read Stage 2 configs") from error
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base_path = cfg.pop("_BASE_", None)
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        cfg = deep_update(load_config(base_path), cfg)
    return cfg


def get_cfg(cfg, path, default=None):
    node = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_rgb(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_rgb(tensor, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def write_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def model_from_config(cfg):
    from .model import GeometryGuidedNAFNet
    model = cfg.get("MODEL", {})
    geometry = cfg.get("GEOMETRY", {})
    return GeometryGuidedNAFNet(
        rgb_channels=model.get("RGB_CHANNELS", 3), geometry_channels=model.get("GEOMETRY_CHANNELS", 5),
        width=model.get("WIDTH", 64), middle_blocks=model.get("MIDDLE_BLOCKS", 8),
        encoder_blocks=tuple(model.get("ENCODER_BLOCKS", [2, 2, 4, 8])),
        decoder_blocks=tuple(model.get("DECODER_BLOCKS", [2, 2, 2, 2])),
        max_residual=model.get("MAX_RESIDUAL", 0.15),
        modulation_scale=geometry.get("MODULATION_SCALE", 0.1),
        confidence_gate=model.get("CONFIDENCE_GATE", False),
        geometry_mode=model.get("GEOMETRY_MODE", "full"),
    )
