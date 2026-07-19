"""Create original/undistorted/displacement/grid QA panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_undistorted_scene import _maps, _scene_layout


def _grid(image: np.ndarray, step: int = 64) -> np.ndarray:
    result = image.copy()
    for x in range(0, result.shape[1], step):
        cv2.line(result, (x, 0), (x, result.shape[0] - 1), (0, 255, 0), 1)
    for y in range(0, result.shape[0], step):
        cv2.line(result, (0, y), (result.shape[1] - 1, y), (0, 255, 0), 1)
    return result


def visualize(source: Path, prepared: Path, output: Path, count: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((prepared / "undistortion_metadata.json").read_text(encoding="utf-8"))
    metadata = next(iter(report["cameras"].values()))
    map_x, map_y = _maps(metadata)
    _, source_images, _, phase1 = _scene_layout(source)
    prepared_images = (prepared / "train" / "images") if phase1 else (prepared / "images")
    names = sorted(path.name for path in source_images.iterdir() if path.is_file())[:count]
    displacement = np.sqrt((map_x - np.arange(map_x.shape[1])[None]) ** 2
                           + (map_y - np.arange(map_y.shape[0])[:, None]) ** 2)
    displacement_vis = cv2.applyColorMap(
        np.clip(displacement / max(float(displacement.max()), 1e-6) * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO)
    valid = ((map_x >= 0) & (map_x < metadata["old_size"][0] - 1)
             & (map_y >= 0) & (map_y < metadata["old_size"][1] - 1)).astype(np.uint8) * 255
    for index, name in enumerate(names):
        original = cv2.imread(str(source_images / name), cv2.IMREAD_COLOR)
        undistorted = cv2.imread(str(prepared_images / name), cv2.IMREAD_COLOR)
        if original is None or undistorted is None:
            continue
        target_size = (undistorted.shape[1], undistorted.shape[0])
        original_resized = cv2.resize(original, target_size)
        panel = np.concatenate((original_resized, undistorted, displacement_vis,
                                cv2.cvtColor(valid, cv2.COLOR_GRAY2BGR), _grid(undistorted)), axis=1)
        cv2.imwrite(str(output / f"{index:03d}_{Path(name).stem}.jpg"), panel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    visualize(args.source.resolve(), args.prepared.resolve(), args.output.resolve(), args.count)
