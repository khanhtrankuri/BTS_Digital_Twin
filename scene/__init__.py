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

import os
import random
import json
from copy import copy
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], optimization_args=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if getattr(args, "camera_use_undistorted_data", False) and getattr(
                args, "camera_require_undistortion_metadata", False):
            metadata_path = os.path.join(args.source_path, "undistortion_metadata.json")
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(
                    "CAMERA.REQUIRE_UNDISTORTION_METADATA is enabled, but the prepared-scene "
                    f"marker is missing: {metadata_path}. Train from the output of "
                    "tools/prepare_undistorted_scene.py, not the raw HCM directory.")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                undistortion_metadata = json.load(handle)
            recorded_output = os.path.normcase(os.path.realpath(undistortion_metadata.get("output", "")))
            current_source = os.path.normcase(os.path.realpath(args.source_path))
            if recorded_output != current_source or int(undistortion_metadata.get("images_processed", 0)) <= 0:
                raise ValueError(
                    "Undistortion metadata does not describe the current non-empty scene output; "
                    "refusing to mix raw images with pinhole intrinsics.")
            print(f"[BTS-GeoGS] Verified prepared undistorted scene: {metadata_path}")

        # Camera construction needs these optimization switches to decide
        # whether edge/geometry priors should be materialized.
        if optimization_args is not None:
            for name in ("geometry_aware", "edge_loss_enabled", "densification_edge_aware",
                         "sky_enabled", "sky_mask_dir", "low_parallax_enabled", "low_parallax_mask_dir",
                         "sharpness_aware_sampling", "resolution_schedule_enabled", "resolution_cache_on_cpu"):
                setattr(args, name, getattr(optimization_args, name, False))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.depths, args.eval, args.train_test_exp,
                validation_split_file=args.validation_split_file,
                strict_sparse_path=args.strict_sparse_path)
        elif os.path.exists(os.path.join(args.source_path, "train", "sparse")) and os.path.exists(os.path.join(args.source_path, "test", "test_poses.csv")):
            print("Found Phase1 train/test dataset layout!")
            scene_info = sceneLoadTypeCallbacks["Phase1"](
                args.source_path, args.images, args.depths, args.eval, args.train_test_exp,
                validation_split_file=args.validation_split_file,
                strict_sparse_path=args.strict_sparse_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            sharpness = [camera.sharpness for camera in self.train_cameras[resolution_scale]]
            if sharpness:
                minimum, maximum = min(sharpness), max(sharpness)
                span = max(maximum - minimum, 1e-12)
                for camera in self.train_cameras[resolution_scale]:
                    camera.normalized_sharpness = (camera.sharpness - minimum) / span
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        self.validation_cameras = None
        if (getattr(args, "validation_full_resolution", False)
                and int(getattr(args, "resolution", 1)) != 1
                and scene_info.test_cameras):
            validation_args = copy(args)
            validation_args.resolution = 1
            validation_args.resolution_schedule_enabled = True
            validation_args.resolution_cache_on_cpu = True
            print("Loading Native-Resolution Validation Cameras")
            self.validation_cameras = cameraList_from_camInfos(
                scene_info.test_cameras,
                1.0,
                validation_args,
                scene_info.is_nerf_synthetic,
                True,
            )

        # One exposure implementation is shared by training, checkpointing and
        # inference. CameraInfo order matches the exposure mapping and Camera uid.
        self.gaussians.setup_exposure(scene_info.train_cameras, optimization_args)
        self.gaussians.setup_background(
            optimization_args, (1.0, 1.0, 1.0) if args.white_background else (0.0, 0.0, 0.0))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
            exposure_path = os.path.join(self.model_path, "exposure.json")
            if self.gaussians.load_exposure_json(exposure_path):
                print("Pretrained diagonal exposures loaded.")
            if self.gaussians.load_background(os.path.join(self.model_path, "background.pt")):
                print("Directional background loaded.")
        else:
            if optimization_args is not None and optimization_args.initialization_mode == "dense_prior":
                from utils.dense_initialization import load_dense_initialization, voxel_downsample
                if not optimization_args.dense_prior_path:
                    raise ValueError("--dense_prior_path is required when --initialization_mode dense_prior is selected.")
                dense_data = load_dense_initialization(optimization_args.dense_prior_path)
                confidence = dense_data.get("confidence")
                if confidence is not None:
                    keep = confidence >= float(optimization_args.dense_prior_confidence_threshold)
                    dense_data = {key: value[keep] if value is not None else None for key, value in dense_data.items()}
                dense_data = voxel_downsample(dense_data, optimization_args.dense_prior_voxel_size)
                self.gaussians.create_from_dense_prior(dense_data, scene_info.train_cameras, self.cameras_extent, optimization_args)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_exposure_json(os.path.join(self.model_path, "exposure.json"))
        self.gaussians.save_background(os.path.join(self.model_path, "background.pt"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getValidationCameras(self):
        if self.validation_cameras is not None:
            return self.validation_cameras
        return self.getTestCameras()
