import json

import numpy as np
import torch
from PIL import Image

from datasets.stage2_dataset import Stage2Dataset


def _config():
    return {
        "MODEL": {"IN_CHANNELS": 10},
        "DATA": {
            "INPUT_COMPONENTS": [
                "gaussian_rgb",
                "normalized_depth",
                "normal",
                "alpha",
                "uncertainty",
                "normalized_variance",
            ],
            "PATCH_SIZE": 4,
            "RANDOM_CROP": True,
            "HORIZONTAL_FLIP": False,
            "VERTICAL_FLIP": False,
            "ROTATE_90": False,
            "USE_SEGMENTATION": False,
        },
        "NORMALIZATION": {
            "ALPHA_THRESHOLD": 0.01,
            "DEPTH_CLIP": 5.0,
            "VARIANCE_CLIP": 5.0,
            "EPS": 1e-6,
            "MIN_VALID_PIXELS": 1,
        },
    }


def _write_dataset(root):
    (root / "rgb").mkdir()
    (root / "gt").mkdir()
    (root / "geometry").mkdir()
    grid = np.arange(36, dtype=np.uint8).reshape(6, 6)
    rgb = np.stack((grid, grid, grid), axis=-1)
    Image.fromarray(rgb, "RGB").save(root / "rgb" / "frame.png")
    Image.fromarray(rgb, "RGB").save(root / "gt" / "frame.png")
    np.savez_compressed(
        root / "geometry" / "frame.npz",
        depth=np.full((1, 6, 6), 2.0, np.float16),
        normal=np.stack(
            (
                np.ones((6, 6), np.float16),
                np.zeros((6, 6), np.float16),
                np.zeros((6, 6), np.float16),
            )
        ),
        alpha=np.ones((1, 6, 6), np.float16),
        uncertainty=np.full((1, 6, 6), 0.25, np.float16),
        depth_variance=np.full((1, 6, 6), 0.04, np.float16),
    )
    frame = {
        "scene": "scene",
        "image_name": "frame.png",
        "rgb_path": "rgb/frame.png",
        "gt_path": "gt/frame.png",
        "geometry_path": "geometry/frame.npz",
        "width": 6,
        "height": 6,
        "split": "train",
        "intrinsics": {"fx": 10, "fy": 10, "cx": 3, "cy": 3},
        "extrinsics": np.eye(4).tolist(),
        "camera_center": [0, 0, 0],
        "view_direction": [0, 0, 1],
    }
    (root / "manifest.json").write_text(
        json.dumps({"frames": [frame]}), encoding="utf-8"
    )


def test_manifest_crop_and_channel_order(tmp_path):
    _write_dataset(tmp_path)
    dataset = Stage2Dataset(
        tmp_path, _config(), split="train", training=True
    )
    sample = dataset[0]
    assert sample["input"].shape == (10, 4, 4)
    assert torch.equal(sample["input"][:3], sample["gt"])
    assert torch.allclose(sample["input"][4], torch.ones(4, 4))
    assert torch.allclose(sample["input"][7], torch.ones(4, 4))
    assert torch.allclose(sample["input"][8], torch.full((4, 4), 0.25))
