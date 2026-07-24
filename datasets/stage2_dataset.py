"""Manifest-backed, geometry-aware Stage-2 dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.stage2_geometry import (
    DEFAULT_COMPONENTS,
    assemble_stage2_input,
    expected_input_channels,
    robust_normalize_geometry,
)
from utils.stage2_io import (
    load_manifest_entries,
    load_rgb_image,
    resolve_entry_path,
)
from utils.stage2_multiview import select_nearest_camera_pairs


_OPTIONAL_MASKS = (
    "foreground_mask",
    "thin_structure_mask",
    "dynamic_mask",
    "sky_background_mask",
)


def _geometry_tensor(
    archive: Any, name: str, channels: int, height: int, width: int
) -> torch.Tensor:
    if name not in archive:
        raise KeyError(f"Geometry archive is missing {name!r}")
    value = torch.from_numpy(np.asarray(archive[name], dtype=np.float32))
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.shape != (channels, height, width):
        raise ValueError(
            f"Geometry {name} must be {(channels, height, width)}, "
            f"got {tuple(value.shape)}"
        )
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class Stage2Dataset(Dataset):
    """Load lossless RGB/GT and float geometry with synchronized transforms."""

    def __init__(
        self,
        manifest_root: str | Path,
        config: dict[str, Any],
        *,
        split: str | None,
        training: bool,
        include_neighbor: bool = False,
        scenes: Sequence[str] | None = None,
    ) -> None:
        all_entries = load_manifest_entries(manifest_root)
        selected_scenes = set(scenes) if scenes is not None else None
        self.entries = [
            entry
            for entry in all_entries
            if (split is None or str(entry.get("split", "train")) == split)
            and (
                selected_scenes is None
                or str(entry.get("scene", "")) in selected_scenes
            )
        ]
        if not self.entries:
            raise ValueError(f"No {split!r} frames found under {manifest_root}")
        data_config = config["DATA"]
        norm_config = config["NORMALIZATION"]
        self.components = tuple(
            data_config.get("INPUT_COMPONENTS", list(DEFAULT_COMPONENTS))
        )
        if bool(data_config.get("USE_SEGMENTATION", False)):
            configured_masks = tuple(
                data_config.get("SEGMENTATION_MASKS", _OPTIONAL_MASKS)
            )
            self.components += tuple(
                name for name in configured_masks if name not in self.components
            )
        expected = expected_input_channels(self.components)
        configured = int(config["MODEL"]["IN_CHANNELS"])
        if expected != configured:
            raise ValueError(
                f"Configured components contain {expected} channels but "
                f"MODEL.IN_CHANNELS={configured}"
            )

        self.training = bool(training)
        self.patch_size = int(data_config.get("PATCH_SIZE", 256))
        self.random_crop = bool(data_config.get("RANDOM_CROP", True)) and self.training
        self.horizontal_flip = (
            bool(data_config.get("HORIZONTAL_FLIP", True)) and self.training
        )
        self.vertical_flip = (
            bool(data_config.get("VERTICAL_FLIP", False)) and self.training
        )
        self.rotate_90 = bool(data_config.get("ROTATE_90", False)) and self.training
        self.include_neighbor = bool(include_neighbor)
        if self.include_neighbor and (
            self.random_crop
            or self.horizontal_flip
            or self.vertical_flip
            or self.rotate_90
        ):
            raise ValueError(
                "Multi-view Stage-2 training requires full, unaugmented images "
                "so exported camera intrinsics remain valid"
            )
        self.normalization = {
            "alpha_threshold": float(norm_config.get("ALPHA_THRESHOLD", 0.01)),
            "depth_clip": float(norm_config.get("DEPTH_CLIP", 5.0)),
            "variance_clip": float(norm_config.get("VARIANCE_CLIP", 5.0)),
            "eps": float(norm_config.get("EPS", 1e-6)),
            "min_valid_pixels": int(norm_config.get("MIN_VALID_PIXELS", 1)),
        }
        self.pairs = (
            select_nearest_camera_pairs(
                self.entries,
                distance_weight=float(
                    data_config.get("PAIR_DISTANCE_WEIGHT", 1.0)
                ),
                direction_weight=float(
                    data_config.get("PAIR_DIRECTION_WEIGHT", 1.0)
                ),
            )
            if self.include_neighbor
            else {}
        )
        if self.include_neighbor and len(self.pairs) != len(self.entries):
            raise ValueError(
                "Every scene needs at least two frames for multi-view training"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_maps(
        self, entry: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        gaussian_rgb = load_rgb_image(resolve_entry_path(entry, "rgb_path"))
        target = load_rgb_image(resolve_entry_path(entry, "gt_path"))
        if gaussian_rgb.shape != target.shape:
            raise ValueError(
                f"RGB/GT size mismatch for {entry.get('image_name')}: "
                f"{tuple(gaussian_rgb.shape)} vs {tuple(target.shape)}"
            )
        _, height, width = gaussian_rgb.shape
        if int(entry.get("width", width)) != width or int(
            entry.get("height", height)
        ) != height:
            raise ValueError(
                f"Manifest dimensions disagree with PNG for {entry.get('image_name')}"
            )

        geometry_path = resolve_entry_path(entry, "geometry_path")
        with np.load(geometry_path, allow_pickle=False) as archive:
            depth = _geometry_tensor(archive, "depth", 1, height, width)
            normal = _geometry_tensor(archive, "normal", 3, height, width)
            alpha = _geometry_tensor(archive, "alpha", 1, height, width).clamp(
                0.0, 1.0
            )
            uncertainty = _geometry_tensor(
                archive, "uncertainty", 1, height, width
            ).clamp(0.0, 1.0)
            if "depth_variance" in archive:
                depth_variance = _geometry_tensor(
                    archive, "depth_variance", 1, height, width
                ).clamp_min(0.0)
                normalized_depth, normalized_variance = robust_normalize_geometry(
                    depth, alpha, depth_variance, **self.normalization
                )
            elif "normalized_variance" in archive:
                normalized_depth, _ = robust_normalize_geometry(
                    depth, alpha, torch.zeros_like(depth), **self.normalization
                )
                normalized_variance = _geometry_tensor(
                    archive, "normalized_variance", 1, height, width
                ).clamp(0.0, 1.0)
                depth_variance = torch.zeros_like(depth)
            else:
                raise KeyError(
                    "Geometry archive requires depth_variance or normalized_variance"
                )
            maps = {
                "gaussian_rgb": gaussian_rgb,
                "normalized_depth": normalized_depth,
                "normal": normal.clamp(-1.0, 1.0),
                "alpha": alpha,
                "uncertainty": uncertainty,
                "normalized_variance": normalized_variance,
                "depth": depth,
                "depth_variance": depth_variance,
            }
            for name in _OPTIONAL_MASKS:
                if name in archive:
                    maps[name] = _geometry_tensor(
                        archive, name, 1, height, width
                    ).clamp(0.0, 1.0)

        for name in self.components:
            if name in _OPTIONAL_MASKS and name not in maps:
                mask_paths = entry.get("mask_paths", {})
                if name not in mask_paths:
                    raise KeyError(
                        f"Configured mask {name!r} is absent for "
                        f"{entry.get('image_name')}"
                    )
                mask_entry = dict(entry)
                mask_entry["_mask_path"] = mask_paths[name]
                mask = load_rgb_image(resolve_entry_path(mask_entry, "_mask_path"))
                maps[name] = mask[:1]
        return maps, target

    @staticmethod
    def _crop(
        maps: dict[str, torch.Tensor],
        target: torch.Tensor,
        top: int,
        left: int,
        size: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        maps = {
            name: value[..., top : top + size, left : left + size]
            for name, value in maps.items()
        }
        return maps, target[..., top : top + size, left : left + size]

    @staticmethod
    def _flip_maps(
        maps: dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        horizontal: bool,
        vertical: bool,
        rotations: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if horizontal:
            maps = {name: value.flip(-1) for name, value in maps.items()}
            target = target.flip(-1)
            maps["normal"][0].mul_(-1.0)
        if vertical:
            maps = {name: value.flip(-2) for name, value in maps.items()}
            target = target.flip(-2)
            maps["normal"][1].mul_(-1.0)
        rotations %= 4
        if rotations:
            maps = {
                name: torch.rot90(value, rotations, (-2, -1))
                for name, value in maps.items()
            }
            target = torch.rot90(target, rotations, (-2, -1))
            normal = maps["normal"]
            for _ in range(rotations):
                old_x, old_y = normal[0].clone(), normal[1].clone()
                normal[0], normal[1] = old_y, -old_x
        return maps, target

    def _prepare(
        self, entry: dict[str, Any], *, augment: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        maps, target = self._load_maps(entry)
        height, width = target.shape[-2:]
        if augment and self.random_crop:
            if min(height, width) < self.patch_size:
                raise ValueError(
                    f"PATCH_SIZE={self.patch_size} exceeds image size "
                    f"{width}x{height} for {entry.get('image_name')}"
                )
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
            maps, target = self._crop(
                maps, target, top, left, self.patch_size
            )
        if augment:
            maps, target = self._flip_maps(
                maps,
                target,
                horizontal=self.horizontal_flip and random.random() < 0.5,
                vertical=self.vertical_flip and random.random() < 0.5,
                rotations=random.randrange(4) if self.rotate_90 else 0,
            )
        return maps, target

    @staticmethod
    def _camera_tensors(entry: dict[str, Any]) -> dict[str, torch.Tensor]:
        intrinsics = entry.get("intrinsics")
        extrinsics = entry.get("extrinsics")
        if not isinstance(intrinsics, dict) or extrinsics is None:
            raise ValueError(
                f"Manifest camera metadata is incomplete for {entry.get('image_name')}"
            )
        return {
            "intrinsics": torch.tensor(
                [
                    intrinsics["fx"],
                    intrinsics["fy"],
                    intrinsics["cx"],
                    intrinsics["cy"],
                ],
                dtype=torch.float32,
            ),
            "extrinsics": torch.tensor(extrinsics, dtype=torch.float32),
        }

    def _sample_dict(
        self,
        maps: dict[str, torch.Tensor],
        target: torch.Tensor,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        stage2_input = assemble_stage2_input(
            maps,
            self.components,
            expected_channels=expected_input_channels(self.components),
        )
        return {
            "input": stage2_input,
            "gt": target,
            "gaussian_rgb": maps["gaussian_rgb"],
            "depth": maps["depth"],
            "alpha": maps["alpha"],
            "uncertainty": maps["uncertainty"],
            "scene": str(entry.get("scene", "")),
            "image_name": str(entry.get("image_name", "")),
            **(
                {"dynamic_mask": maps["dynamic_mask"]}
                if "dynamic_mask" in self.components
                else {}
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        maps, target = self._prepare(entry, augment=self.training)
        sample = self._sample_dict(maps, target, entry)
        if self.include_neighbor:
            neighbor_entry = self.entries[self.pairs[index]]
            neighbor_maps, neighbor_target = self._prepare(
                neighbor_entry, augment=False
            )
            neighbor = self._sample_dict(
                neighbor_maps, neighbor_target, neighbor_entry
            )
            for key in (
                "input",
                "gt",
                "gaussian_rgb",
                "depth",
                "alpha",
                "uncertainty",
                "scene",
                "image_name",
            ):
                sample[f"neighbor_{key}"] = neighbor[key]
            if "dynamic_mask" in neighbor:
                sample["neighbor_dynamic_mask"] = neighbor["dynamic_mask"]
            sample.update(
                {
                    f"camera_{key}": value
                    for key, value in self._camera_tensors(entry).items()
                }
            )
            sample.update(
                {
                    f"neighbor_camera_{key}": value
                    for key, value in self._camera_tensors(neighbor_entry).items()
                }
            )
        return sample
