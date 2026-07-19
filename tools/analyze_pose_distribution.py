"""Report train/validation/test pose coverage and difficulty bins."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pose_utils import (center_direction_from_qvec_tvec, difficulty_bin,
                             nearest_pose, scene_radius)
from utils.read_write_model import read_model


def analyze(source: Path, split: Path | None = None) -> dict:
    base = source / "train" if (source / "train" / "sparse" / "0").is_dir() else source
    cameras, images, _ = read_model(str(base / "sparse" / "0"))
    available = {path.name for path in (base / "images").iterdir() if path.is_file()}
    train_records = [image for image in images.values() if image.name in available]
    holdout_records = []
    if split is not None:
        split_data = json.loads(split.read_text(encoding="utf-8"))
        train_names = set(split_data["train_images"])
        holdout_names = set(split_data["validation_images"])
        holdout_records = [image for image in train_records if image.name in holdout_names]
        train_records = [image for image in train_records if image.name in train_names]
    train_pose = [center_direction_from_qvec_tvec(image.qvec, image.tvec) for image in train_records]
    centers = np.stack([value[0] for value in train_pose])
    directions = np.stack([value[1] for value in train_pose])
    radius = scene_radius(centers)

    targets = []
    for image in holdout_records:
        center, direction = center_direction_from_qvec_tvec(image.qvec, image.tvec)
        targets.append((image.name, center, direction))
    test_csv = source / "test" / "test_poses.csv"
    if split is None and test_csv.exists():
        with test_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                qvec = [float(row[key]) for key in ("qw", "qx", "qy", "qz")]
                tvec = [float(row[key]) for key in ("tx", "ty", "tz")]
                center, direction = center_direction_from_qvec_tvec(qvec, tvec)
                targets.append((row["image_name"], center, direction))
    samples = []
    train_names = [image.name for image in train_records]
    for name, center, direction in targets:
        nearest, position, angle = nearest_pose(center, direction, centers, directions, radius)
        samples.append({"image_name": name, "nearest_train_camera": train_names[nearest],
                        "normalized_position_distance": position, "view_angle_degrees": angle,
                        "difficulty_bin": difficulty_bin(position, angle)})
    position = np.asarray([sample["normalized_position_distance"] for sample in samples])
    angle = np.asarray([sample["view_angle_degrees"] for sample in samples])
    summary = {
        "source": str(source.resolve()), "train_count": len(train_records), "target_count": len(samples),
        "scene_radius": radius, "difficulty_counts": dict(Counter(sample["difficulty_bin"] for sample in samples)),
        "position": ({"median": float(np.median(position)), "p90": float(np.quantile(position, 0.9)),
                      "max": float(position.max())} if position.size else {}),
        "angle_degrees": ({"median": float(np.median(angle)), "p90": float(np.quantile(angle, 0.9)),
                           "max": float(angle.max())} if angle.size else {}),
        "samples": samples,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = analyze(options.source, options.split)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
