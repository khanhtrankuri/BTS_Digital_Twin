"""Prepare a pinhole COLMAP scene from distorted training imagery.

The source is never modified.  The script supports both a normal COLMAP layout
(``images``, ``sparse/0``) and the competition Phase-1 layout
(``train/images``, ``train/sparse/0``).  Feature observations are transformed
with the same map as the images; extrinsics and 3D points stay unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.camera_models import camera_matrix, opencv_distortion, parse_colmap_intrinsics
from utils.read_write_model import Camera, Image, read_model, write_model


def _scene_layout(root: Path) -> tuple[Path, Path, Path, bool]:
    if (root / "train" / "sparse" / "0").is_dir():
        return root / "train", root / "train" / "images", root / "train" / "sparse" / "0", True
    if (root / "sparse" / "0").is_dir():
        return root, root / "images", root / "sparse" / "0", False
    raise FileNotFoundError(f"No sparse/0 COLMAP model found below {root}")


def _new_camera(camera: Camera, alpha: float, crop_mode: str) -> tuple[Camera, dict]:
    parsed = parse_colmap_intrinsics(camera.model, camera.params)
    old_k = camera_matrix(parsed.fx, parsed.fy, parsed.cx, parsed.cy)
    dist = opencv_distortion(parsed.camera_model, parsed.distortion)
    size = (int(camera.width), int(camera.height))
    if not np.any(np.abs(dist) > 0):
        new_k, roi = old_k.copy(), (0, 0, size[0], size[1])
    else:
        new_k, roi = cv2.getOptimalNewCameraMatrix(old_k, dist, size, float(alpha), size)
    x, y, width, height = map(int, roi)
    if crop_mode == "valid" and width > 0 and height > 0:
        new_k = new_k.copy()
        new_k[0, 2] -= x
        new_k[1, 2] -= y
        output_size = (width, height)
        crop_roi = (x, y, width, height)
    else:
        output_size = size
        crop_roi = (0, 0, size[0], size[1])
    params = np.array([new_k[0, 0], new_k[1, 1], new_k[0, 2], new_k[1, 2]], dtype=np.float64)
    converted = Camera(id=camera.id, model="PINHOLE", width=output_size[0], height=output_size[1], params=params)
    metadata = {
        "camera_id": int(camera.id), "old_model": parsed.camera_model,
        "old_size": list(size), "new_size": list(output_size),
        "old_camera_matrix": old_k.tolist(), "old_distortion": list(parsed.distortion),
        "opencv_distortion": dist.tolist(), "new_camera_matrix": new_k.tolist(),
        "crop_roi": list(crop_roi),
    }
    return converted, metadata


def _maps(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    old_size = tuple(metadata["old_size"])
    crop_x, crop_y, width, height = metadata["crop_roi"]
    old_k = np.asarray(metadata["old_camera_matrix"], dtype=np.float64)
    dist = np.asarray(metadata["opencv_distortion"], dtype=np.float64)
    new_k = np.asarray(metadata["new_camera_matrix"], dtype=np.float64).copy()
    # The stored matrix is already shifted for a valid crop. OpenCV needs the
    # pre-crop matrix to construct the full-size map.
    new_k[0, 2] += crop_x
    new_k[1, 2] += crop_y
    map_x, map_y = cv2.initUndistortRectifyMap(
        old_k, dist, None, new_k, old_size, cv2.CV_32FC1)
    if (crop_x, crop_y, width, height) != (0, 0, old_size[0], old_size[1]):
        map_x = map_x[crop_y:crop_y + height, crop_x:crop_x + width]
        map_y = map_y[crop_y:crop_y + height, crop_x:crop_x + width]
    return map_x, map_y


def _remap(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, kind: str) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if kind in {"mask", "depth"} else cv2.INTER_LINEAR
    result = cv2.remap(array, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)
    if kind == "normal":
        values = result.astype(np.float32)
        if values.ndim == 3 and values.shape[2] >= 3:
            normal = values[..., :3]
            encoded = np.issubdtype(array.dtype, np.integer) or normal.max() > 1.5
            if encoded:
                normal = normal / 127.5 - 1.0
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6)
            if encoded:
                normal = np.clip((normal + 1.0) * 127.5, 0, 255)
            values[..., :3] = normal
        result = values.astype(array.dtype)
    return result


def redistortion_maps(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map pixels on the original distorted grid into the prepared pinhole grid."""

    old_width, old_height = map(int, metadata["old_size"])
    old_k = np.asarray(metadata["old_camera_matrix"], dtype=np.float64)
    distortion = np.asarray(metadata["opencv_distortion"], dtype=np.float64)
    new_k = np.asarray(metadata["new_camera_matrix"], dtype=np.float64)
    grid_x, grid_y = np.meshgrid(
        np.arange(old_width, dtype=np.float32),
        np.arange(old_height, dtype=np.float32))
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(
        distorted_pixels, old_k, distortion, P=new_k).reshape(old_height, old_width, 2)
    return undistorted_pixels[..., 0], undistorted_pixels[..., 1]


def redistort_render(array: np.ndarray, metadata: dict) -> np.ndarray:
    """Resample a prepared-grid RGB render onto the source/evaluator pixel grid."""

    new_width, new_height = map(int, metadata["new_size"])
    if array.shape[:2] != (new_height, new_width):
        raise ValueError(
            f"Expected an undistorted render of {new_width}x{new_height}, "
            f"got {array.shape[1]}x{array.shape[0]}")
    map_x, map_y = redistortion_maps(metadata)
    return cv2.remap(
        array, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"OpenCV could not read {path}")
    return array


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, array)
    elif not cv2.imwrite(str(path), array):
        raise IOError(f"OpenCV could not write {path}")


def _transform_observations(image: Image, old_camera: Camera, metadata: dict) -> Image:
    if image.xys.size == 0:
        return image
    parsed = parse_colmap_intrinsics(old_camera.model, old_camera.params)
    old_k = camera_matrix(parsed.fx, parsed.fy, parsed.cx, parsed.cy)
    dist = opencv_distortion(parsed.camera_model, parsed.distortion)
    new_k = np.asarray(metadata["new_camera_matrix"], dtype=np.float64)
    xys = cv2.undistortPoints(
        np.asarray(image.xys, dtype=np.float64).reshape(-1, 1, 2), old_k, dist, P=new_k).reshape(-1, 2)
    return Image(id=image.id, qvec=image.qvec, tvec=image.tvec, camera_id=image.camera_id,
                 name=image.name, xys=xys, point3D_ids=image.point3D_ids)


def _update_test_csv(source_root: Path, output_root: Path, camera_metadata: dict[int, dict]) -> None:
    source_csv = source_root / "test" / "test_poses.csv"
    if not source_csv.exists() or not camera_metadata:
        return
    output_test = output_root / "test"
    output_test.mkdir(parents=True, exist_ok=True)
    metadata = next(iter(camera_metadata.values()))
    matrix = np.asarray(metadata["new_camera_matrix"])
    with source_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else []
    for row in rows:
        row.update({"width": str(metadata["new_size"][0]), "height": str(metadata["new_size"][1]),
                    "fx": repr(float(matrix[0, 0])), "fy": repr(float(matrix[1, 1])),
                    "cx": repr(float(matrix[0, 2])), "cy": repr(float(matrix[1, 2]))})
    with (output_test / "test_poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_scene(args: argparse.Namespace) -> dict:
    source_root, output_root = Path(args.source).resolve(), Path(args.output).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output must be a separate directory outside the source tree")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    input_base, image_dir, sparse_dir, phase1 = _scene_layout(source_root)
    output_base = output_root / "train" if phase1 else output_root
    output_images, output_sparse = output_base / "images", output_base / "sparse" / "0"
    output_images.mkdir(parents=True, exist_ok=True)
    output_sparse.mkdir(parents=True, exist_ok=True)

    model = read_model(str(sparse_dir))
    if model is None:
        raise ValueError(f"Could not read COLMAP model at {sparse_dir}")
    cameras, images, points3d = model
    converted_cameras: dict[int, Camera] = {}
    metadata: dict[int, dict] = {}
    map_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for camera_id, camera in cameras.items():
        converted_cameras[camera_id], metadata[camera_id] = _new_camera(camera, args.alpha, args.crop_mode)
        map_cache[camera_id] = _maps(metadata[camera_id])

    converted_images: dict[int, Image] = {}
    for image_id, image in images.items():
        source_image = image_dir / image.name
        if not source_image.exists():
            converted_images[image_id] = _transform_observations(image, cameras[image.camera_id], metadata[image.camera_id])
            continue
        array = _read_array(source_image)
        map_x, map_y = map_cache[image.camera_id]
        _write_array(output_images / image.name, _remap(array, map_x, map_y, "image"))
        converted_images[image_id] = _transform_observations(image, cameras[image.camera_id], metadata[image.camera_id])

    prior_specs: list[tuple[str, str]] = []
    if args.process_depth:
        prior_specs += [("depth", "depth"), ("depths", "depth")]
    if args.process_normal:
        prior_specs += [("normal", "normal"), ("normals", "normal")]
    if args.process_masks:
        prior_specs += [("masks", "mask"), ("confidence", "mask"), ("edge", "mask"), ("alpha", "mask")]
    image_by_stem = {Path(image.name).stem: image for image in images.values()}
    for relative_dir, kind in prior_specs:
        source_dir = input_base / relative_dir
        if not source_dir.is_dir():
            continue
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            image = image_by_stem.get(path.stem)
            if image is None:
                continue
            array = _read_array(path)
            map_x, map_y = map_cache[image.camera_id]
            _write_array(output_base / relative_dir / path.relative_to(source_dir), _remap(array, map_x, map_y, kind))

    output_points = points3d if args.copy_sparse else {}
    write_model(converted_cameras, converted_images, output_points, str(output_sparse), ext=".bin")
    if phase1:
        _update_test_csv(source_root, output_root, metadata)
    report = {
        "source": str(source_root), "output": str(output_root), "phase1_layout": phase1,
        "alpha": args.alpha, "crop_mode": args.crop_mode,
        "cameras": {str(key): value for key, value in metadata.items()},
        "images_processed": sum((image_dir / image.name).exists() for image in images.values()),
        "points3D_copied": len(output_points),
        "warning": ("The updated test CSV describes the undistorted pixel grid. If the evaluator expects "
                    "distorted RGB, render in this grid and explicitly redistort the final image." if phase1 else ""),
    }
    metadata_path = output_root / "undistortion_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--crop_mode", choices=("valid", "same"), default="valid")
    parser.add_argument("--copy_sparse", action="store_true", help="Copy points3D in addition to cameras/images.")
    parser.add_argument("--process_depth", action="store_true")
    parser.add_argument("--process_normal", action="store_true")
    parser.add_argument("--process_masks", action="store_true")
    return parser


if __name__ == "__main__":
    result = prepare_scene(build_parser().parse_args())
    print(json.dumps(result, indent=2))
