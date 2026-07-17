"""Manifest-backed Stage 2 dataset with geometry-consistent patch augmentation."""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .geometry import preprocess_geometry
from .utils import load_rgb


def _absolute(manifest_dir, path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(manifest_dir, path))


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError("manifest must be a list or contain a 'samples' list")
    return samples


def assert_disjoint_splits(samples):
    identities = {}
    for sample in samples:
        identity = (sample.get("scene", ""), str(sample.get("camera_id", sample.get("image_name", ""))))
        split = sample.get("split")
        previous = identities.setdefault(identity, split)
        if previous != split:
            raise ValueError(f"Data leakage: sample {identity} occurs in both {previous} and {split}")


class Stage2RefinementDataset(Dataset):
    def __init__(self, manifest_path, split, patch_size=None, augment=True, cache_mode="none",
                 edge_patch_ratio=0.30, high_residual_patch_ratio=0.20,
                 depth_normalization="robust_per_view", alpha_threshold=0.01,
                 vertical_flip=True, rotate90=True):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        all_samples = load_manifest(self.manifest_path)
        assert_disjoint_splits(all_samples)
        self.samples = [sample for sample in all_samples if sample.get("split") == split]
        if not self.samples:
            raise ValueError(f"No samples with split='{split}' in {manifest_path}")
        self.split = split
        self.patch_size = int(patch_size) if patch_size else None
        self.augment = bool(augment and split == "train")
        self.cache_mode = cache_mode
        self.edge_patch_ratio = float(edge_patch_ratio)
        self.high_residual_patch_ratio = float(high_residual_patch_ratio)
        self.depth_normalization = depth_normalization
        self.alpha_threshold = alpha_threshold
        self.vertical_flip = vertical_flip
        self.rotate90 = rotate90
        self._cache = {}
        if cache_mode not in ("none", "ram"):
            raise ValueError("cache_mode must be 'none' or 'ram'")

    def __len__(self):
        return len(self.samples)

    def _load(self, index):
        if index in self._cache:
            return {key: value.clone() if torch.is_tensor(value) else value for key, value in self._cache[index].items()}
        sample = self.samples[index]
        rgb = load_rgb(_absolute(self.manifest_dir, sample["rgb_render"]))
        target = load_rgb(_absolute(self.manifest_dir, sample["rgb_gt"]))
        depth = torch.from_numpy(np.load(_absolute(self.manifest_dir, sample["depth"]))).float()
        normal = torch.from_numpy(np.load(_absolute(self.manifest_dir, sample["normal"]))).float()
        alpha = torch.from_numpy(np.load(_absolute(self.manifest_dir, sample["alpha"]))).float()
        if depth.ndim == 2: depth = depth.unsqueeze(0)
        if alpha.ndim == 2: alpha = alpha.unsqueeze(0)
        if normal.ndim == 3 and normal.shape[-1] == 3: normal = normal.permute(2, 0, 1)
        shapes = {tuple(x.shape[-2:]) for x in (rgb, target, depth, normal, alpha)}
        if len(shapes) != 1:
            raise ValueError(f"Shape mismatch in sample {sample.get('image_name', index)}: {shapes}")
        depth, normal, alpha, valid = preprocess_geometry(
            depth, normal, alpha, self.depth_normalization, self.alpha_threshold,
            scene_scale=sample.get("scene_scale"))
        item = {"rgb": rgb.clamp(0, 1), "depth": depth, "normal": normal,
                "alpha": alpha, "target": target.clamp(0, 1), "depth_valid": valid,
                "scene": sample.get("scene", ""), "camera_id": sample.get("camera_id", index),
                "image_name": sample.get("image_name", str(index))}
        if self.cache_mode == "ram":
            self._cache[index] = {key: value.clone() if torch.is_tensor(value) else value for key, value in item.items()}
        return item

    @staticmethod
    def _sobel_score(image):
        gray = image.mean(0, keepdim=True).unsqueeze(0)
        kx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, kx.transpose(-1, -2), padding=1)
        return (gx.square() + gy.square()).sqrt()[0, 0]

    def _crop_origin(self, item, size):
        height, width = item["rgb"].shape[-2:]
        max_y, max_x = height - size, width - size
        if max_y <= 0 and max_x <= 0:
            return 0, 0
        draw = random.random()
        score = None
        if draw < self.high_residual_patch_ratio:
            score = (item["rgb"] - item["target"]).abs().mean(0)
        elif draw < self.high_residual_patch_ratio + self.edge_patch_ratio:
            score = self._sobel_score(item["target"])
        if score is None:
            return random.randint(0, max(0, max_y)), random.randint(0, max(0, max_x))
        candidates = [(random.randint(0, max(0, max_y)), random.randint(0, max(0, max_x))) for _ in range(12)]
        return max(candidates, key=lambda p: score[p[0]:p[0]+size, p[1]:p[1]+size].mean().item())

    def _crop(self, item):
        if not self.patch_size:
            return item
        size = self.patch_size
        height, width = item["rgb"].shape[-2:]
        pad_h, pad_w = max(0, size - height), max(0, size - width)
        if pad_h or pad_w:
            for key in ("rgb", "depth", "normal", "alpha", "target", "depth_valid"):
                value = item[key].float() if item[key].dtype == torch.bool else item[key]
                item[key] = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate")
            item["depth_valid"] = item["depth_valid"].bool()
        y, x = self._crop_origin(item, size)
        for key in ("rgb", "depth", "normal", "alpha", "target", "depth_valid"):
            item[key] = item[key][..., y:y+size, x:x+size]
        return item

    def _augment(self, item):
        keys = ("rgb", "depth", "normal", "alpha", "target", "depth_valid")
        if random.random() < 0.5:
            for key in keys: item[key] = torch.flip(item[key], dims=(-1,))
            item["normal"][0].neg_()
        if self.vertical_flip and random.random() < 0.5:
            for key in keys: item[key] = torch.flip(item[key], dims=(-2,))
            item["normal"][1].neg_()
        if self.rotate90:
            turns = random.randint(0, 3)
            if turns:
                for key in keys: item[key] = torch.rot90(item[key], turns, dims=(-2, -1))
                nx, ny = item["normal"][0].clone(), item["normal"][1].clone()
                if turns == 1:
                    item["normal"][0], item["normal"][1] = -ny, nx
                elif turns == 2:
                    item["normal"][0], item["normal"][1] = -nx, -ny
                else:
                    item["normal"][0], item["normal"][1] = ny, -nx
        return item

    def __getitem__(self, index):
        item = self._crop(self._load(index))
        return self._augment(item) if self.augment else item
