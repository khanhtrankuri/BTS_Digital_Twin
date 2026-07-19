r"""
Render test poses from `test_poses.csv` for original-style 3D Gaussian Splatting repos.

Put this file inside the root of your gaussian-splatting repo, next to `train.py`.

Example:
python render_test_poses_3dgs.py ^
  -s "C:\path\to\scene\train" ^
  -m "C:\path\to\outputs\scene" ^
  --test_csv "C:\path\to\scene\test\test_poses.csv" ^
  --out_dir "C:\path\to\submission\scene"
"""

from pathlib import Path
from argparse import ArgumentParser
import csv

import numpy as np
import torch
import torchvision
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov


def qvec2rotmat(qvec):
    """
    COLMAP quaternion format: qw, qx, qy, qz.
    Returns world-to-camera rotation matrix.
    """
    q = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero quaternion")
    q = q / norm
    w, x, y, z = q

    return np.array([
        [1 - 2 * y * y - 2 * z * z,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [    2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z,     2 * y * z - 2 * x * w],
        [    2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def row_to_float(row, key):
    return float(row[key])


def row_to_int(row, key):
    return int(float(row[key]))


def make_camera_from_csv_row(row, uid, data_device="cuda"):
    image_name = row["image_name"]

    width = row_to_int(row, "width")
    height = row_to_int(row, "height")

    fx = row_to_float(row, "fx")
    fy = row_to_float(row, "fy")
    cx = row_to_float(row, "cx")
    cy = row_to_float(row, "cy")

    qw = row_to_float(row, "qw")
    qx = row_to_float(row, "qx")
    qy = row_to_float(row, "qy")
    qz = row_to_float(row, "qz")

    tx = row_to_float(row, "tx")
    ty = row_to_float(row, "ty")
    tz = row_to_float(row, "tz")

    # COLMAP pose convention:
    # x_cam = R_w2c * x_world + T_w2c
    R_w2c = qvec2rotmat([qw, qx, qy, qz])
    T_w2c = np.array([tx, ty, tz], dtype=np.float32)

    # Original 3DGS loader stores R as transpose of COLMAP R.
    # Camera.getWorld2View2 later transposes it back internally.
    R_for_3dgs = R_w2c.T.astype(np.float32)

    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    # Camera class requires an image to know width/height. For test camera,
    # use a dummy black PIL image and keep the CSV intrinsics.
    dummy_image = Image.new("RGB", (width, height))

    cam = Camera(
        resolution=(width, height),
        colmap_id=uid,
        R=R_for_3dgs,
        T=T_w2c,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=dummy_image,
        invdepthmap=None,
        image_name=image_name,
        uid=uid,
        cx=cx,
        cy=cy,
        source_width=width,
        source_height=height,
        data_device=data_device,
        has_ground_truth=False,
    )
    return cam


def read_test_poses(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = [
        "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
        "fx", "fy", "cx", "cy", "width", "height"
    ]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing columns in test_poses.csv: {missing}")

    return rows


def main():
    parser = ArgumentParser(description="Render images from test_poses.csv using 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--exposure_compensation", action="store_true")
    parser.add_argument("--test_exposure_mode", choices=["identity", "nearest_camera", "weighted_nearest",
                        "pose_confidence_blend", "temporal_weighted", "pose_temporal_weighted"], default="identity")

    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("3DGS rendering requires CUDA. torch.cuda.is_available() is False.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_test_poses(args.test_csv)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    print(f"Loaded model from: {args.model_path}")
    print(f"Rendering {len(rows)} test poses")
    print(f"Output dir: {out_dir}")

    with torch.no_grad():
        for uid, row in enumerate(rows):
            image_name = row["image_name"]
            out_path = out_dir / image_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_path.exists():
                print(f"[SKIP] {image_name}")
                continue

            cam = make_camera_from_csv_row(row, uid=uid, data_device=device)
            pkg = render(cam, gaussians, pipe, background,
                         apply_exposure=args.exposure_compensation,
                         exposure_mode=args.test_exposure_mode)
            rendering = pkg["render"].clamp(0.0, 1.0)

            torchvision.utils.save_image(rendering, str(out_path))
            print(f"[OK] {image_name}  {cam.image_width}x{cam.image_height}")

            del cam, pkg, rendering
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
