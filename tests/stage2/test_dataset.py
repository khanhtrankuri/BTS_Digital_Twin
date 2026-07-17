import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from stage2_refiner.dataset import Stage2RefinementDataset, assert_disjoint_splits
from stage2_refiner.trainer import train


def make_manifest(tmp_path):
    samples = []
    for index, split in enumerate(("train", "val")):
        rgb = np.full((12, 14, 3), 64 + index, np.uint8); Image.fromarray(rgb).save(tmp_path / f"rgb{index}.png"); Image.fromarray(rgb).save(tmp_path / f"gt{index}.png")
        np.save(tmp_path / f"d{index}.npy", np.ones((1, 12, 14), np.float32))
        normal = np.zeros((3, 12, 14), np.float32); normal[2] = 1; np.save(tmp_path / f"n{index}.npy", normal)
        np.save(tmp_path / f"a{index}.npy", np.ones((1, 12, 14), np.float32))
        samples.append({"scene": "S", "camera_id": index, "image_name": f"{index}.png", "split": split,
                        "rgb_render": f"rgb{index}.png", "rgb_gt": f"gt{index}.png", "depth": f"d{index}.npy",
                        "normal": f"n{index}.npy", "alpha": f"a{index}.npy"})
    path = tmp_path / "manifest.json"; path.write_text(json.dumps({"samples": samples}), encoding="utf-8"); return path


def test_dataset_shape_range_and_split(tmp_path):
    dataset = Stage2RefinementDataset(make_manifest(tmp_path), "train", patch_size=8, augment=False)
    item = dataset[0]
    assert item["rgb"].shape == item["target"].shape == (3, 8, 8)
    assert item["depth"].shape == item["alpha"].shape == (1, 8, 8)
    assert torch.isfinite(item["depth"]).all() and 0 <= item["rgb"].min() <= item["rgb"].max() <= 1
    assert torch.allclose(torch.linalg.vector_norm(item["normal"], dim=0), torch.ones(8, 8))


def test_horizontal_flip_updates_normal_x(tmp_path):
    dataset = Stage2RefinementDataset(make_manifest(tmp_path), "train", augment=True, vertical_flip=False, rotate90=False)
    item = dataset._load(0); item["normal"][0].fill_(.5)
    with patch("stage2_refiner.dataset.random.random", return_value=0.0): result = dataset._augment(item)
    assert torch.all(result["normal"][0] == -.5)


def test_leakage_detection():
    with pytest.raises(ValueError, match="Data leakage"):
        assert_disjoint_splits([{"scene": "S", "camera_id": 1, "split": "train"}, {"scene": "S", "camera_id": 1, "split": "val"}])


def test_one_epoch_trainer_and_resume(tmp_path):
    manifest = make_manifest(tmp_path)
    cfg = {"MODEL": {"WIDTH": 8, "ENCODER_BLOCKS": [1], "DECODER_BLOCKS": [1], "MIDDLE_BLOCKS": 1,
                     "MAX_RESIDUAL": .15, "GEOMETRY_MODE": "full"},
           "DATA": {"PATCH_SIZE": 8, "BATCH_SIZE": 1, "VAL_BATCH_SIZE": 1, "NUM_WORKERS": 0, "AUGMENT": False},
           "LOSS": {"MSE_WEIGHT": 1.0}, "OPTIMIZER": {"LR": .001, "WEIGHT_DECAY": 0.0},
           "SCHEDULER": {"WARMUP_STEPS": 0, "MIN_LR": 1e-6},
           "TRAIN": {"EPOCHS": 1, "AMP": False, "EMA": True, "SAVE_EVERY": 1, "SEED": 1,
                     "EARLY_STOPPING_PATIENCE": 0}, "EVALUATION": {"PSNR_MAX": 30.0}}
    output = tmp_path / "out"; first = train(cfg, str(manifest), str(output), device="cpu")
    assert first["step"] == 1 and (output / "best.pth").exists()
    cfg["TRAIN"]["EPOCHS"] = 2
    resumed = train(cfg, str(manifest), str(output), resume=str(output / "last.pth"), device="cpu")
    assert resumed["step"] == 2
