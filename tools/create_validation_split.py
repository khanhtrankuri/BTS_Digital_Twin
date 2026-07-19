"""Create deterministic pose-aware validation splits without using test RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pose_utils import (center_direction_from_qvec_tvec, difficulty_bin,
                             nearest_pose, scene_radius)
from utils.read_write_model import read_model


def _paths(source: Path) -> tuple[Path, Path]:
    base = source / "train" if (source / "train" / "sparse" / "0").is_dir() else source
    sparse = base / "sparse" / "0"
    images = base / "images"
    if not sparse.is_dir() or not images.is_dir():
        raise FileNotFoundError(f"Expected images and sparse/0 below {base}")
    return sparse, images


def _natural_key(name: str):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def _select_indices(mode: str, names: list[str], centers: np.ndarray,
                    directions: np.ndarray, count: int, temporal_offset: int) -> list[int]:
    if mode == "temporal_matched":
        order = sorted(range(len(names)), key=lambda index: _natural_key(names[index]))
        stride = max(2, round(len(names) / count))
        selected = order[temporal_offset % stride::stride]
        if len(selected) < count:
            selected += [index for index in order if index not in selected][:count - len(selected)]
        return sorted(selected[:count])
    if mode == "position_extrapolation":
        score = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    elif mode == "angular_extrapolation":
        mean_direction = directions.mean(axis=0)
        mean_direction /= max(np.linalg.norm(mean_direction), 1e-12)
        score = np.arccos(np.clip(directions @ mean_direction, -1.0, 1.0))
    else:
        raise ValueError(f"Unknown split mode {mode}")
    return sorted(np.argsort(score, kind="stable")[-count:].tolist())


def create_split(source: Path, output: Path, mode: str, ratio: float,
                 temporal_offset: int) -> dict:
    sparse_path, image_path = _paths(source)
    model = read_model(str(sparse_path))
    if model is None:
        raise ValueError(f"Could not read COLMAP model at {sparse_path}")
    _, images, _ = model
    records = [image for image in images.values() if (image_path / image.name).exists()]
    records.sort(key=lambda image: _natural_key(image.name))
    if len(records) < 3:
        raise ValueError("At least three training images are required")
    names = [image.name for image in records]
    pose = [center_direction_from_qvec_tvec(image.qvec, image.tvec) for image in records]
    centers = np.stack([value[0] for value in pose])
    directions = np.stack([value[1] for value in pose])
    count = min(len(records) - 2, max(1, int(round(len(records) * ratio))))
    holdout_indices = _select_indices(mode, names, centers, directions, count, temporal_offset)
    holdout_set = set(holdout_indices)
    train_indices = [index for index in range(len(records)) if index not in holdout_set]
    radius = scene_radius(centers[train_indices])
    samples = []
    for index in holdout_indices:
        nearest_index, position, angle = nearest_pose(
            centers[index], directions[index], centers[train_indices], directions[train_indices], radius)
        source_index = train_indices[nearest_index]
        samples.append({
            "image_name": names[index], "nearest_train_camera": names[source_index],
            "normalized_position_distance": position, "view_angle_degrees": angle,
            "difficulty_bin": difficulty_bin(position, angle),
        })
    report = {
        "version": 1, "source": str(source.resolve()), "protocol": "diagnostic_full_sparse",
        "split_mode": mode, "holdout_ratio": ratio, "scene_radius": radius,
        "train_images": [names[index] for index in train_indices],
        "validation_images": [names[index] for index in holdout_indices],
        "samples": samples,
        "warning": "This split still references the original sparse reconstruction. Use rebuild_train_only_colmap.py for strict validation.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "train.txt").write_text("\n".join(report["train_images"]) + "\n", encoding="utf-8")
    (output / "test.txt").write_text("\n".join(report["validation_images"]) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("temporal_matched", "position_extrapolation", "angular_extrapolation"),
                        default="temporal_matched")
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--temporal_offset", type=int, default=0)
    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    if not 0.0 < options.ratio < 1.0:
        raise ValueError("--ratio must be between 0 and 1")
    result = create_split(options.source, options.output, options.mode, options.ratio, options.temporal_offset)
    summary = {key: value for key, value in result.items()
               if key not in {"samples", "train_images", "validation_images"}}
    summary.update({"train_count": len(result["train_images"]),
                    "validation_count": len(result["validation_images"])})
    print(json.dumps(summary, indent=2))
