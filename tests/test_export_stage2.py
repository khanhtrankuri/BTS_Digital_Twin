import numpy as np
from PIL import Image

from tools.export_stage2_dataset import (
    stage2_files_complete,
    validate_geometry_arrays,
)


def _arrays(height=3, width=4):
    return {
        "depth": np.ones((1, height, width), np.float32),
        "normal": np.zeros((3, height, width), np.float32),
        "alpha": np.ones((1, height, width), np.float32),
        "uncertainty": np.zeros((1, height, width), np.float32),
        "depth_variance": np.zeros((1, height, width), np.float32),
    }


def test_export_geometry_is_finite_and_shaped():
    arrays = _arrays()
    validate_geometry_arrays(arrays, 3, 4)
    arrays["depth"][0, 0, 0] = np.nan
    try:
        validate_geometry_arrays(arrays, 3, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN geometry should be rejected")


def test_export_resume_detects_complete_files(tmp_path):
    rgb = tmp_path / "frame.png"
    gt = tmp_path / "frame_gt.png"
    geometry = tmp_path / "frame.npz"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), "RGB").save(rgb)
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), "RGB").save(gt)
    np.savez_compressed(geometry, **_arrays())
    assert stage2_files_complete(rgb, gt, geometry)
    geometry.unlink()
    assert not stage2_files_complete(rgb, gt, geometry)
