from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from tools.prepare_undistorted_scene import prepare_scene
from utils.read_write_model import Camera, Image, read_model, write_model


def test_offline_undistortion_writes_pinhole_model_without_touching_source(tmp_path):
    source = tmp_path / "source"
    sparse = source / "sparse" / "0"
    images_dir = source / "images"
    sparse.mkdir(parents=True)
    images_dir.mkdir()
    camera = Camera(1, "SIMPLE_RADIAL", 8, 6, np.array([6.0, 4.0, 3.0, 0.01]))
    image_record = Image(1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "frame.png",
                         np.array([[2.0, 2.0]]), np.array([-1], dtype=np.int64))
    write_model({1: camera}, {1: image_record}, {}, str(sparse), ext=".bin")
    original = np.zeros((6, 8, 3), dtype=np.uint8)
    original[2:4, 3:5] = 255
    assert cv2.imwrite(str(images_dir / "frame.png"), original)

    output = tmp_path / "output"
    prepare_scene(Namespace(source=str(source), output=str(output), alpha=0.0,
                            crop_mode="same", copy_sparse=True, process_depth=False,
                            process_normal=False, process_masks=False))
    converted_camera = read_model(str(output / "sparse" / "0"))[0][1]
    assert converted_camera.model == "PINHOLE"
    assert (source / "sparse" / "0" / "cameras.bin").exists()
    assert (output / "undistortion_metadata.json").exists()
