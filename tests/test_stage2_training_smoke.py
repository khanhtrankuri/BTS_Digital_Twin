import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def _write_frame(root: Path, index: int, split: str) -> dict:
    stem = f"frame_{index}"
    array = np.full((12, 12, 3), 80 + index * 10, dtype=np.uint8)
    Image.fromarray(array, "RGB").save(root / "rgb" / f"{stem}.png")
    Image.fromarray(array, "RGB").save(root / "gt" / f"{stem}.png")
    np.savez_compressed(
        root / "geometry" / f"{stem}.npz",
        depth=np.full((1, 12, 12), 2.0, np.float16),
        normal=np.zeros((3, 12, 12), np.float16),
        alpha=np.ones((1, 12, 12), np.float16),
        uncertainty=np.zeros((1, 12, 12), np.float16),
        depth_variance=np.zeros((1, 12, 12), np.float16),
    )
    return {
        "scene": "smoke",
        "image_name": f"{stem}.png",
        "rgb_path": f"rgb/{stem}.png",
        "gt_path": f"gt/{stem}.png",
        "geometry_path": f"geometry/{stem}.npz",
        "width": 12,
        "height": 12,
        "split": split,
        "intrinsics": {"fx": 10, "fy": 10, "cx": 6, "cy": 6},
        "extrinsics": np.eye(4).tolist(),
        "camera_center": [float(index), 0, 0],
        "view_direction": [0, 0, 1],
    }


def test_train_stage2_one_step_writes_finite_checkpoint(tmp_path):
    data_root = tmp_path / "data" / "smoke"
    for name in ("rgb", "gt", "geometry"):
        (data_root / name).mkdir(parents=True, exist_ok=True)
    frames = [
        _write_frame(data_root, 0, "train"),
        _write_frame(data_root, 1, "val"),
    ]
    (data_root / "manifest.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    config = {
        "STAGE2": {"ENABLED": False},
        "MODEL": {
            "NAME": "geonaf",
            "IN_CHANNELS": 10,
            "OUT_CHANNELS": 4,
            "WIDTH": 4,
            "ENC_BLOCKS": [1],
            "MIDDLE_BLOCKS": 1,
            "DEC_BLOCKS": [1],
            "RESIDUAL_SCALE": 0.15,
            "UNCERTAINTY_GATING": False,
            "UNCERTAINTY_CHANNEL_INDEX": 8,
            "MIN_MASK": 0.05,
        },
        "DATA": {
            "INPUT_COMPONENTS": [
                "gaussian_rgb",
                "normalized_depth",
                "normal",
                "alpha",
                "uncertainty",
                "normalized_variance",
            ],
            "PATCH_SIZE": 8,
            "RANDOM_CROP": True,
            "HORIZONTAL_FLIP": False,
            "VERTICAL_FLIP": False,
            "ROTATE_90": False,
            "NUM_WORKERS": 0,
            "SEED": 7,
            "USE_SEGMENTATION": False,
        },
        "NORMALIZATION": {
            "ALPHA_THRESHOLD": 0.01,
            "DEPTH_CLIP": 5.0,
            "VARIANCE_CLIP": 5.0,
            "EPS": 1e-6,
            "MIN_VALID_PIXELS": 1,
        },
        "LOSS": {
            "CHARBONNIER": 1.0,
            "SSIM": 0.2,
            "EDGE": 0.1,
            "PERCEPTUAL": 0.0,
            "RESIDUAL": 0.01,
            "MASK": 0.005,
            "IDENTITY": 0.05,
            "MULTIVIEW": 0.0,
        },
        "TRAIN": {
            "OPTIMIZER": "AdamW",
            "EPOCHS": 1,
            "BATCH_SIZE": 1,
            "LR": 2e-4,
            "WEIGHT_DECAY": 1e-4,
            "AMP": False,
            "GRAD_CLIP": 1.0,
            "EARLY_STOPPING": 0,
        },
        "INFERENCE": {"TILE_SIZE": 0, "TILE_OVERLAP": 0},
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_dir = tmp_path / "output"
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "train_stage2.py"),
            "--config",
            str(config_path),
            "--manifest_root",
            str(tmp_path / "data"),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--max_train_steps",
            "1",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(
        output_dir / "latest.pth", map_location="cpu", weights_only=False
    )
    assert checkpoint["global_step"] == 1
    assert all(
        torch.isfinite(value).all()
        for value in checkpoint["model"].values()
        if torch.is_floating_point(value)
    )
