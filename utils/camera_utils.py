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

import numpy as np
import os
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import torch
from typing import TYPE_CHECKING
from utils.camera_models import validate_camera_intrinsics, warn_if_pinhole_approximation

if TYPE_CHECKING:
    from scene.cameras import Camera

WARNED = False
_MISSING_PRIOR_WARNINGS = set()
_WARNED_DISTORTION_MODELS = set()


def _find_prior(root, image_name):
    if not root:
        return None
    stem = os.path.splitext(os.path.basename(image_name))[0]
    for extension in (".npy", ".npz", ".pt", ".pth", ".png", ".tif", ".tiff"):
        candidate = os.path.join(root, stem + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_prior(path, kind):
    if path is None:
        return None
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        array = archive[kind] if kind in archive else archive[archive.files[0]]
    elif suffix in (".pt", ".pth"):
        value = torch.load(path, map_location="cpu", weights_only=False)
        array = value[kind] if isinstance(value, dict) and kind in value else value
        array = array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array)
    else:
        array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    array = np.asarray(array, dtype=np.float32)
    if kind == "mask" and array.ndim == 3:
        array = array[..., :3].max(axis=-1)
        if array.max() > 1.5:
            array = array / 255.0
    if kind == "normal" and array.ndim == 3 and array.shape[-1] == 3:
        array = array.transpose(2, 0, 1)
        # Common encoded PNG normal convention.
        if array.max() > 1.5:
            array = array / 127.5 - 1.0
    elif array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array.transpose(2, 0, 1)
    return array


def _camera_prior(args, image_name, directory_attr, kind):
    directory = getattr(args, directory_attr, "")
    if directory and not os.path.isabs(directory):
        direct = os.path.join(args.source_path, directory)
        phase1 = os.path.join(args.source_path, "train", directory)
        directory = phase1 if not os.path.isdir(direct) and os.path.isdir(phase1) else direct
    path = _find_prior(directory, image_name)
    if directory and path is None and directory_attr not in _MISSING_PRIOR_WARNINGS:
        print(f"[BTS-GeoGS] No {kind} prior matched training images under '{directory}'; skipping it.")
        _MISSING_PRIOR_WARNINGS.add(directory_attr)
    return _load_prior(path, kind) if path else None

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    from scene.cameras import Camera

    has_ground_truth = bool(cam_info.image_path) and cam_info.image_path != "" and os.path.exists(cam_info.image_path)
    if has_ground_truth:
        image = Image.open(cam_info.image_path)
    else:
        image = Image.new("RGB", (cam_info.width, cam_info.height))

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    depth_prior = _camera_prior(args, cam_info.image_name, "depth_prior_dir", "depth")
    normal_prior = _camera_prior(args, cam_info.image_name, "normal_prior_dir", "normal")
    confidence_map = _camera_prior(args, cam_info.image_name, "confidence_prior_dir", "confidence")
    sky_mask = (_camera_prior(args, cam_info.image_name, "sky_mask_dir", "mask")
                if getattr(args, "sky_enabled", False) else None)
    low_parallax_mask = (_camera_prior(args, cam_info.image_name, "low_parallax_mask_dir", "mask")
                         if getattr(args, "low_parallax_enabled", False) else None)
    validate_camera_intrinsics(cam_info, strict=getattr(args, "camera_strict_model_validation", True))
    camera_model = getattr(cam_info, "camera_model", "PINHOLE")
    if getattr(args, "camera_use_undistorted_data", False) and camera_model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError("CAMERA.USE_UNDISTORTED_DATA is true, but the scene still contains a distorted camera model")
    if (getattr(args, "camera_warn_if_distortion_dropped", True)
            and camera_model not in _WARNED_DISTORTION_MODELS
            and warn_if_pinhole_approximation(cam_info)):
        _WARNED_DISTORTION_MODELS.add(camera_model)
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  cx=getattr(cam_info, "cx", None), cy=getattr(cam_info, "cy", None),
                  camera_model=camera_model, fx=getattr(cam_info, "fx", None), fy=getattr(cam_info, "fy", None),
                  distortion=getattr(cam_info, "distortion_params", ()),
                  difficulty_bin=getattr(cam_info, "difficulty_bin", ""),
                  normalized_position_distance=getattr(cam_info, "normalized_position_distance", 0.0),
                  view_angle_degrees=getattr(cam_info, "view_angle_degrees", 0.0),
                  source_width=cam_info.width, source_height=cam_info.height,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  has_ground_truth=has_ground_truth, depth_prior=depth_prior,
                  normal_prior=normal_prior, confidence_map=confidence_map,
                  sky_mask=sky_mask, low_parallax_mask=low_parallax_mask,
                  compute_sharpness=getattr(args, "sharpness_aware_sampling", False),
                  cache_images_on_cpu=(getattr(args, "resolution_schedule_enabled", False)
                                       and getattr(args, "resolution_cache_on_cpu", True)),
                  compute_local_sharpness=(getattr(args, "sharpness_aware_sampling", False)
                                           or getattr(args, "densification_edge_aware", False)),
                  compute_edge=getattr(args, "geometry_aware", False) and
                  (getattr(args, "edge_loss_enabled", False) or getattr(args, "densification_edge_aware", False)))

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera):
    fov_y = camera.FoVy if hasattr(camera, "FoVy") else camera.FovY
    fov_x = camera.FoVx if hasattr(camera, "FoVx") else camera.FovX
    cx = getattr(camera, "cx", None)
    cy = getattr(camera, "cy", None)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(fov_y, camera.height),
        'fx' : fov2focal(fov_x, camera.width),
        'cx' : cx if cx is not None else camera.width / 2.0,
        'cy' : cy if cy is not None else camera.height / 2.0,
        'camera_model': getattr(camera, "camera_model", "PINHOLE"),
        'distortion': (camera.distortion.detach().cpu().tolist()
                       if getattr(camera, "distortion", None) is not None
                       else list(getattr(camera, "distortion_params", ()))),
    }
    return camera_entry
