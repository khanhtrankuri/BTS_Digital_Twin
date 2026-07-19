"""Map renders from an undistorted PINHOLE grid back to the raw image grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def redistort(image: np.ndarray, metadata: dict) -> np.ndarray:
    old_width, old_height = metadata["old_size"]
    old_k = np.asarray(metadata["old_camera_matrix"], dtype=np.float64)
    distortion = np.asarray(metadata["opencv_distortion"], dtype=np.float64)
    new_k = np.asarray(metadata["new_camera_matrix"], dtype=np.float64)
    v, u = np.meshgrid(np.arange(old_height, dtype=np.float32),
                       np.arange(old_width, dtype=np.float32), indexing="ij")
    distorted_pixels = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(
        distorted_pixels, old_k, distortion, P=new_k).reshape(old_height, old_width, 2)
    return cv2.remap(image, undistorted_pixels[..., 0], undistorted_pixels[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--camera_id", type=int)
    args = parser.parse_args()
    report = json.loads(args.metadata.read_text(encoding="utf-8"))
    cameras = report["cameras"]
    camera_id = str(args.camera_id) if args.camera_id is not None else next(iter(cameras))
    camera_metadata = cameras[camera_id]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.iterdir()):
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            cv2.imwrite(str(args.output / path.name), redistort(image, camera_metadata))
