#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import os
import shutil
import subprocess
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="colmap", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="magick", type=str)
    return parser


def run_command(command, error_message):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logging.error("%s failed with code %s. Exiting.", error_message, exc.returncode)
        raise SystemExit(exc.returncode) from exc


def main():
    args = build_parser().parse_args()
    use_gpu = "0" if args.no_gpu else "1"

    if not args.skip_matching:
        os.makedirs(os.path.join(args.source_path, "distorted", "sparse"), exist_ok=True)

        run_command([
            args.colmap_executable,
            "feature_extractor",
            "--database_path", os.path.join(args.source_path, "distorted", "database.db"),
            "--image_path", os.path.join(args.source_path, "input"),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", args.camera,
            "--SiftExtraction.use_gpu", use_gpu,
        ], "Feature extraction")

        run_command([
            args.colmap_executable,
            "exhaustive_matcher",
            "--database_path", os.path.join(args.source_path, "distorted", "database.db"),
            "--SiftMatching.use_gpu", use_gpu,
        ], "Feature matching")

        run_command([
            args.colmap_executable,
            "mapper",
            "--database_path", os.path.join(args.source_path, "distorted", "database.db"),
            "--image_path", os.path.join(args.source_path, "input"),
            "--output_path", os.path.join(args.source_path, "distorted", "sparse"),
            "--Mapper.ba_global_function_tolerance=0.000001",
        ], "Mapper")

    run_command([
        args.colmap_executable,
        "image_undistorter",
        "--image_path", os.path.join(args.source_path, "input"),
        "--input_path", os.path.join(args.source_path, "distorted", "sparse", "0"),
        "--output_path", args.source_path,
        "--output_type", "COLMAP",
    ], "Image undistortion")

    sparse_dir = os.path.join(args.source_path, "sparse")
    sparse_zero_dir = os.path.join(sparse_dir, "0")
    os.makedirs(sparse_zero_dir, exist_ok=True)
    for file_name in os.listdir(sparse_dir):
        if file_name == "0":
            continue
        shutil.move(os.path.join(sparse_dir, file_name), os.path.join(sparse_zero_dir, file_name))

    if args.resize:
        print("Copying and resizing...")

        for scale_dir in ["images_2", "images_4", "images_8"]:
            os.makedirs(os.path.join(args.source_path, scale_dir), exist_ok=True)

        for file_name in os.listdir(os.path.join(args.source_path, "images")):
            source_file = os.path.join(args.source_path, "images", file_name)

            resize_jobs = [
                ("images_2", "50%"),
                ("images_4", "25%"),
                ("images_8", "12.5%"),
            ]
            for target_dir, resize in resize_jobs:
                destination_file = os.path.join(args.source_path, target_dir, file_name)
                shutil.copy2(source_file, destination_file)
                run_command([
                    args.magick_executable,
                    "mogrify",
                    "-resize", resize,
                    destination_file,
                ], f"{resize} resize")

    print("Done.")


if __name__ == "__main__":
    main()
