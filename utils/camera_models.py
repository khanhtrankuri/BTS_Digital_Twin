"""Camera-model parsing, validation, and scaling utilities.

COLMAP extrinsics in this repository are world-to-camera transforms.  Runtime
``Camera.R`` is stored transposed for the row-vector convention used by the
rasterizer, while the intrinsic values in this module use ordinary pixel
coordinates with centers at integer-plus-one-half locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import warnings

import numpy as np


PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}
RADIAL_MODELS = {"SIMPLE_RADIAL", "RADIAL"}
OPENCV_MODELS = {"OPENCV", "FULL_OPENCV"}
SUPPORTED_CAMERA_MODELS = PINHOLE_MODELS | RADIAL_MODELS | OPENCV_MODELS


@dataclass(frozen=True)
class ParsedIntrinsics:
    """Normalized intrinsic representation independent of COLMAP layout."""

    camera_model: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]


def parse_colmap_intrinsics(camera_model: str, params: Iterable[float]) -> ParsedIntrinsics:
    """Parse supported COLMAP parameters without discarding distortion."""

    model = str(camera_model).upper()
    values = np.asarray(tuple(params), dtype=np.float64)
    expected = {
        "SIMPLE_PINHOLE": 3,
        "PINHOLE": 4,
        "SIMPLE_RADIAL": 4,
        "RADIAL": 5,
        "OPENCV": 8,
        "FULL_OPENCV": 12,
    }
    if model not in expected:
        raise ValueError(f"Unsupported COLMAP camera model: {camera_model}")
    if values.size != expected[model]:
        raise ValueError(f"{model} expects {expected[model]} parameters, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError(f"{model} contains non-finite intrinsic parameters")

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = float(values[0])
        cx, cy = map(float, values[1:3])
        distortion = tuple(float(v) for v in values[3:])
    else:
        fx, fy, cx, cy = map(float, values[:4])
        distortion = tuple(float(v) for v in values[4:])
    return ParsedIntrinsics(model, fx, fy, cx, cy, distortion)


def opencv_distortion(camera_model: str, distortion: Iterable[float]) -> np.ndarray:
    """Convert stored coefficients to OpenCV's pinhole distortion vector."""

    model = str(camera_model).upper()
    values = tuple(float(v) for v in distortion)
    if model in PINHOLE_MODELS:
        return np.zeros(5, dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        return np.array([values[0], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "RADIAL":
        return np.array([values[0], values[1], 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV":
        # COLMAP and OpenCV both use k1, k2, p1, p2 here.
        return np.array([values[0], values[1], values[2], values[3], 0.0], dtype=np.float64)
    if model == "FULL_OPENCV":
        # k1, k2, p1, p2, k3, k4, k5, k6.
        return np.asarray(values[:8], dtype=np.float64)
    raise ValueError(f"Cannot build an OpenCV distortion vector for {camera_model}")


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Return a conventional 3x3 pixel-space intrinsic matrix."""

    return np.array([[float(fx), 0.0, float(cx)],
                     [0.0, float(fy), float(cy)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_intrinsics(fx: float, fy: float, cx: float, cy: float,
                     source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Scale non-centered intrinsics between arbitrary image resolutions."""

    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive when scaling intrinsics")
    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)
    return float(fx) * sx, float(fy) * sy, float(cx) * sx, float(cy) * sy


def _read(camera: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(camera, name):
            value = getattr(camera, name)
            if value is not None:
                return value
    return default


def validate_camera_intrinsics(camera: Any, strict: bool = True) -> list[str]:
    """Validate a ``CameraInfo`` or runtime ``Camera``.

    Returns warnings in non-strict mode and raises ``ValueError`` in strict
    mode. Principal points may be off-center, but must stay within a generous
    one-image margin to catch swapped dimensions and corrupted metadata.
    """

    model = str(_read(camera, "camera_model", default="PINHOLE")).upper()
    width = int(_read(camera, "width", "image_width", default=0))
    height = int(_read(camera, "height", "image_height", default=0))
    fx = float(_read(camera, "fx", default=0.0))
    fy = float(_read(camera, "fy", default=0.0))
    cx = float(_read(camera, "cx", default=np.nan))
    cy = float(_read(camera, "cy", default=np.nan))
    distortion = _read(camera, "distortion_params", "distortion", default=())
    if distortion is None:
        distortion_values = np.empty(0, dtype=np.float64)
    else:
        try:
            import torch
            if torch.is_tensor(distortion):
                distortion = distortion.detach().cpu().numpy()
        except ImportError:
            pass
        distortion_values = np.asarray(distortion, dtype=np.float64).reshape(-1)

    errors: list[str] = []
    if model not in SUPPORTED_CAMERA_MODELS:
        errors.append(f"unsupported camera model {model}")
    if width <= 0 or height <= 0:
        errors.append(f"invalid resolution {width}x{height}")
    if not np.isfinite([fx, fy]).all() or fx <= 0.0 or fy <= 0.0:
        errors.append(f"invalid focal lengths fx={fx}, fy={fy}")
    if not np.isfinite([cx, cy]).all():
        errors.append("principal point is non-finite")
    elif width > 0 and height > 0 and not (-width <= cx <= 2 * width and -height <= cy <= 2 * height):
        errors.append(f"principal point ({cx}, {cy}) is implausible for {width}x{height}")
    if not np.isfinite(distortion_values).all():
        errors.append("distortion coefficients contain NaN or Inf")
    expected_distortion = {"SIMPLE_PINHOLE": 0, "PINHOLE": 0, "SIMPLE_RADIAL": 1,
                           "RADIAL": 2, "OPENCV": 4, "FULL_OPENCV": 8}.get(model)
    if expected_distortion is not None and distortion_values.size != expected_distortion:
        errors.append(f"{model} expects {expected_distortion} distortion coefficients, got {distortion_values.size}")
    if errors and strict:
        raise ValueError("Invalid camera intrinsics: " + "; ".join(errors))
    return errors


def warn_if_pinhole_approximation(camera: Any, enabled: bool = True) -> bool:
    """Emit the required non-silent warning when a distorted camera is rendered."""

    if not enabled:
        return False
    model = str(_read(camera, "camera_model", default="PINHOLE")).upper()
    distortion = _read(camera, "distortion_params", "distortion", default=())
    if distortion is None:
        return False
    try:
        import torch
        if torch.is_tensor(distortion):
            distortion = distortion.detach().cpu().numpy()
    except ImportError:
        pass
    values = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if model in RADIAL_MODELS | OPENCV_MODELS and values.size and np.any(np.abs(values) > 0.0):
        warnings.warn(
            f"{model} camera is being rendered as PINHOLE. This can cause several-pixel "
            "projection errors near image boundaries. Prepare an undistorted scene or disable "
            "this warning only after validating the approximation.", RuntimeWarning, stacklevel=2)
        return True
    return False
