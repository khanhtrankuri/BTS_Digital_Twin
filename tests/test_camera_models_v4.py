from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from utils.camera_models import (opencv_distortion, parse_colmap_intrinsics,
                                 scale_intrinsics, validate_camera_intrinsics)
from scene.cameras import Camera


def test_simple_radial_parsing_preserves_k1():
    parsed = parse_colmap_intrinsics("SIMPLE_RADIAL", [925.0, 660.0, 494.5, 0.009])
    assert parsed.fx == parsed.fy == 925.0
    assert parsed.distortion == (0.009,)
    assert np.allclose(opencv_distortion(parsed.camera_model, parsed.distortion),
                       [0.009, 0.0, 0.0, 0.0, 0.0])


def test_intrinsics_scale_non_centered_principal_point():
    scaled = scale_intrinsics(1000.0, 900.0, 610.0, 430.0, (1200, 900), (600, 300))
    assert scaled == pytest.approx((500.0, 300.0, 305.0, 143.3333333333))


def test_strict_intrinsics_validation_rejects_bad_focal():
    camera = SimpleNamespace(camera_model="PINHOLE", width=640, height=480,
                             fx=-1.0, fy=500.0, cx=320.0, cy=240.0, distortion=())
    with pytest.raises(ValueError, match="focal"):
        validate_camera_intrinsics(camera)


def test_progressive_camera_cache_keeps_images_on_cpu():
    camera = Camera(
        (16, 8), 1, np.eye(3), np.zeros(3), 1.0, 0.8, None,
        Image.new("RGB", (32, 16)), None, "frame_001.png", 0,
        cx=15.0, cy=7.0, source_width=32, source_height=16,
        camera_model="PINHOLE", fx=20.0, fy=18.0, distortion=(),
        data_device="cpu", cache_images_on_cpu=True)
    assert camera.original_image.device.type == "cpu"
    assert camera.fx == pytest.approx(10.0)
    assert camera.cx == pytest.approx(7.5)
