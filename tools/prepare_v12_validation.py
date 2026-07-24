"""Create leaderboard-shaped, multi-scene validation folds for BTS v12.

Only train RGB and test *poses/names* are used. For each scene, the validation
fraction is chosen so that validation/train has the same ratio as the private
test/train set. Temporal interleaving matches the way the private frames are
withheld from these capture sequences more closely than a single extreme-pose
holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.create_validation_split import create_split


SCENES = (
    "bonsai",
    "chair",
    "HCM0421",
    "HCM0539",
    "HCM0540",
    "HCM0644",
    "HCM0674",
)


def _source_for(scene: str, data_root: Path, prepared_root: Path) -> Path:
    return (prepared_root if scene.startswith("HCM") else data_root) / scene


def _train_image_count(source: Path) -> int:
    base = source / "train" if (source / "train" / "images").is_dir() else source
    image_dir = base / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    return sum(path.is_file() for path in image_dir.iterdir())


def _test_pose_count(source: Path) -> int:
    path = source / "test" / "test_poses.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def prepare(
    data_root: Path,
    prepared_root: Path,
    output_root: Path,
    folds: int,
    scenes: list[str],
    reuse: bool,
) -> dict:
    if folds < 1:
        raise ValueError("--folds must be positive")
    manifest = {
        "version": 1,
        "protocol": "native_resolution_temporal_matched",
        "leakage_policy": (
            "test RGB is never read; test pose count sets only the holdout ratio; "
            "the original sparse model remains diagnostic, not strict"
        ),
        "folds": [],
    }
    for fold in range(folds):
        fold_record = {"fold": fold, "scenes": {}}
        for scene in scenes:
            source = _source_for(scene, data_root, prepared_root).resolve()
            train_count = _train_image_count(source)
            test_count = _test_pose_count(source)
            holdout_ratio = test_count / float(train_count + test_count)
            destination = output_root / f"fold_{fold}" / scene
            split_path = destination / "validation_split.json"
            if split_path.is_file() and reuse:
                report = json.loads(split_path.read_text(encoding="utf-8"))
            else:
                if destination.exists() and any(destination.iterdir()):
                    raise FileExistsError(
                        f"Refusing to overwrite non-empty validation directory: {destination}")
                report = create_split(
                    source,
                    destination,
                    mode="temporal_matched",
                    ratio=holdout_ratio,
                    temporal_offset=fold,
                )
            fold_record["scenes"][scene] = {
                "source": str(source),
                "split": str(split_path.resolve()),
                "train_images": len(report["train_images"]),
                "validation_images": len(report["validation_images"]),
                "private_test_poses": test_count,
                "holdout_ratio": holdout_ratio,
                "difficulty_counts": dict(
                    Counter(sample["difficulty_bin"] for sample in report["samples"])
                ),
            }
        manifest["folds"].append(fold_record)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Lenovo\Documents\Val_Race"),
    )
    parser.add_argument(
        "--prepared_root",
        type=Path,
        default=REPO_ROOT / "data" / "bts_v12_prepared",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "data" / "bts_v12_validation",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--reuse", action="store_true")
    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    result = prepare(
        options.data_root,
        options.prepared_root,
        options.output_root,
        options.folds,
        options.scenes,
        options.reuse,
    )
    print(json.dumps(result, indent=2))
