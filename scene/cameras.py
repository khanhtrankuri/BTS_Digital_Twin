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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.camera_models import scale_intrinsics, validate_camera_intrinsics
from utils.general_utils import PILtoTorch
import cv2
import torch.nn.functional as F
from utils.exposure_utils import frame_time_from_name

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 cx=None, cy=None, source_width=None, source_height=None,
                 camera_model="PINHOLE", fx=None, fy=None, distortion=None,
                 difficulty_bin="", normalized_position_distance=0.0, view_angle_degrees=0.0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 has_ground_truth = True, depth_prior=None, normal_prior=None,
                 confidence_map=None, sky_mask=None, low_parallax_mask=None,
                 compute_edge=False, compute_sharpness=False, compute_local_sharpness=False,
                 cache_images_on_cpu=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.view_index = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.camera_model = str(camera_model).upper()
        self.cx = cx
        self.cy = cy
        self.source_width = source_width
        self.source_height = source_height
        self.image_name = image_name
        self.frame_time = frame_time_from_name(image_name)
        self.difficulty_bin = str(difficulty_bin)
        self.normalized_position_distance = float(normalized_position_distance)
        self.view_angle_degrees = float(view_angle_degrees)
        self.has_ground_truth = has_ground_truth

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        self.cache_images_on_cpu = bool(cache_images_on_cpu)
        self.image_storage_device = torch.device("cpu") if self.cache_images_on_cpu else self.data_device
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.image_storage_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.image_storage_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.image_storage_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.width = self.image_width
        self.height = self.image_height

        source_width = float(self.source_width or self.image_width)
        source_height = float(self.source_height or self.image_height)
        source_fx = float(fx if fx is not None and fx > 0 else fov2focal(self.FoVx, source_width))
        source_fy = float(fy if fy is not None and fy > 0 else fov2focal(self.FoVy, source_height))
        source_cx = float(self.cx if self.cx is not None else source_width / 2.0)
        source_cy = float(self.cy if self.cy is not None else source_height / 2.0)
        self.fx, self.fy, self.cx, self.cy = scale_intrinsics(
            source_fx, source_fy, source_cx, source_cy,
            (int(source_width), int(source_height)), (self.image_width, self.image_height))
        if distortion is None or len(distortion) == 0:
            self.distortion = None
        else:
            self.distortion = torch.as_tensor(distortion, dtype=torch.float32, device=self.data_device).reshape(-1)

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.image_storage_device)

        def resize_prior(prior, channels, mode):
            if prior is None:
                return None
            prior = torch.as_tensor(prior, dtype=torch.float32, device=self.image_storage_device)
            if prior.ndim == 2:
                prior = prior.unsqueeze(0)
            if prior.ndim != 3 or prior.shape[0] not in (1, channels):
                raise ValueError(f"Expected a [C,H,W] geometry prior with C={channels}.")
            prior = prior.unsqueeze(0)
            kwargs = {"align_corners": False} if mode != "nearest" else {}
            return F.interpolate(prior, size=(self.image_height, self.image_width), mode=mode, **kwargs).squeeze(0)

        self.depth_prior = resize_prior(depth_prior, 1, "bilinear")
        self.normal_prior = resize_prior(normal_prior, 3, "bilinear")
        if self.normal_prior is not None:
            self.normal_prior = F.normalize(torch.nan_to_num(self.normal_prior), dim=0, eps=1e-6)
        self.confidence_map = resize_prior(confidence_map, 1, "bilinear")
        if self.confidence_map is not None:
            self.confidence_map = torch.clamp(torch.nan_to_num(self.confidence_map), 0.0, 1.0)
        self.sky_mask = resize_prior(sky_mask, 1, "nearest")
        self.low_parallax_mask = resize_prior(low_parallax_mask, 1, "nearest")
        if self.sky_mask is not None:
            self.sky_mask = (torch.nan_to_num(self.sky_mask) > 0.5).float()
        if self.low_parallax_mask is not None:
            self.low_parallax_mask = (torch.nan_to_num(self.low_parallax_mask) > 0.5).float()
        self.edge_map = None
        if compute_edge:
            from utils.geometry_losses import sobel_edge_map
            self.edge_map = sobel_edge_map(self.original_image)
        self.sharpness = 1.0
        self.normalized_sharpness = 1.0
        self.local_sharpness = None
        if compute_sharpness or compute_local_sharpness:
            gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            self.sharpness = float(np.var(laplacian))
            if compute_local_sharpness:
                local = np.abs(laplacian)
                scale_value = max(float(np.quantile(local, 0.95)), 1e-6)
                local = np.clip(local / scale_value, 0.0, 1.0)
                self.local_sharpness = resize_prior(local, 1, "bilinear")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        validate_camera_intrinsics(self)
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale), dtype=torch.float32,
            device=self.data_device).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.image_width,
            height=self.image_height,
        ).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

