"""Build and audit a train-only COLMAP model for strict validation.

``--mode colmap`` is the strict protocol: it invokes COLMAP's
``point_triangulator`` with the original feature database and only train poses.
``--mode filter_tracks`` is a diagnostic fallback; it removes holdout tracks
but cannot undo their past influence on already-triangulated XYZ coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import Image, Point3D, read_model, write_model


def _numeric_leaves(value, prefix=""):
    leaves = {}
    if isinstance(value, dict):
        for key, child in value.items():
            leaves.update(_numeric_leaves(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)):
        leaves[prefix] = float(value)
    return leaves


def _layout(source: Path) -> tuple[Path, Path]:
    base = source / "train" if (source / "train" / "sparse" / "0").is_dir() else source
    return base / "sparse" / "0", base / "images"


def _load_split(path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data["train_images"]), set(data["validation_images"])


def _filter_model(cameras, images, points3d, train_names: set[str], min_track_length: int):
    train_images = {key: value for key, value in images.items() if value.name in train_names}
    train_ids = set(train_images)
    filtered_points = {}
    for point_id, point in points3d.items():
        keep = np.array([int(image_id) in train_ids for image_id in point.image_ids], dtype=bool)
        if int(keep.sum()) < min_track_length:
            continue
        filtered_points[point_id] = Point3D(
            id=point.id, xyz=point.xyz, rgb=point.rgb, error=point.error,
            image_ids=point.image_ids[keep], point2D_idxs=point.point2D_idxs[keep])
    kept_ids = set(filtered_points)
    filtered_images = {}
    for image_id, image in train_images.items():
        point_ids = np.asarray(image.point3D_ids).copy()
        point_ids[~np.isin(point_ids, list(kept_ids))] = -1
        filtered_images[image_id] = Image(
            id=image.id, qvec=image.qvec, tvec=image.tvec, camera_id=image.camera_id,
            name=image.name, xys=image.xys, point3D_ids=point_ids)
    camera_ids = {image.camera_id for image in filtered_images.values()}
    filtered_cameras = {key: value for key, value in cameras.items() if key in camera_ids}
    return filtered_cameras, filtered_images, filtered_points


def rebuild(args: argparse.Namespace) -> dict:
    sparse_path, image_path = _layout(args.source.resolve())
    cameras, images, points3d = read_model(str(sparse_path))
    train_names, holdout_names = _load_split(args.split)
    holdout_ids = {image_id for image_id, image in images.items() if image.name in holdout_names}
    points_with_holdout = 0
    holdout_observations = 0
    for point in points3d.values():
        hits = sum(int(image_id) in holdout_ids for image_id in point.image_ids)
        points_with_holdout += int(hits > 0)
        holdout_observations += hits

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    filtered = _filter_model(cameras, images, points3d, train_names, args.min_track_length)
    strict = args.mode == "colmap"
    if args.mode == "filter_tracks":
        write_model(*filtered, str(output), ext=".bin")
    else:
        if args.database is None or not args.database.is_file():
            raise FileNotFoundError("--database is required for strict COLMAP retriangulation")
        executable = shutil.which(args.colmap)
        if executable is None:
            raise FileNotFoundError(f"COLMAP executable not found: {args.colmap}")
        seed = output / "train_pose_seed"
        seed.mkdir()
        empty_images = {
            key: Image(id=value.id, qvec=value.qvec, tvec=value.tvec, camera_id=value.camera_id,
                       name=value.name, xys=np.empty((0, 2)), point3D_ids=np.empty((0,), dtype=np.int64))
            for key, value in filtered[1].items()
        }
        write_model(filtered[0], empty_images, {}, str(seed), ext=".bin")
        strict_output = output / "sparse"
        strict_output.mkdir()
        command = [executable, "point_triangulator", "--database_path", str(args.database.resolve()),
                   "--image_path", str(image_path.resolve()), "--input_path", str(seed),
                   "--output_path", str(strict_output)]
        subprocess.run(command, check=True)
        shutil.rmtree(seed)

    resulting_points = len(read_model(str(output / "sparse" if strict else output))[2])
    report = {
        "protocol": "strict_train_only_sparse" if strict else "diagnostic_filtered_tracks",
        "source_points": len(points3d), "result_points": resulting_points,
        "holdout_images": len(holdout_ids), "points_seen_by_holdout": points_with_holdout,
        "holdout_track_observations": holdout_observations,
        "point_count_difference": len(points3d) - resulting_points,
        "strict": strict,
        "warning": ("" if strict else
                    "Filtered tracks are not a strict retriangulation because original XYZ may have used holdout observations."),
    }
    if args.diagnostic_metrics and args.strict_metrics:
        diagnostic = _numeric_leaves(json.loads(args.diagnostic_metrics.read_text(encoding="utf-8")))
        strict_values = _numeric_leaves(json.loads(args.strict_metrics.read_text(encoding="utf-8")))
        report["metric_gap_strict_minus_diagnostic"] = {
            key: strict_values[key] - diagnostic[key]
            for key in sorted(diagnostic.keys() & strict_values.keys())
        }
    (output / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path, help="validation_split.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("colmap", "filter_tracks"), default="colmap")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--min_track_length", type=int, default=2)
    parser.add_argument("--diagnostic_metrics", type=Path)
    parser.add_argument("--strict_metrics", type=Path)
    return parser


if __name__ == "__main__":
    print(json.dumps(rebuild(build_parser().parse_args()), indent=2))
