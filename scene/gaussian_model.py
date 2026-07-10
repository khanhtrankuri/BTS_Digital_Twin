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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.residual_accum = torch.empty(0)
        self.edge_accum = torch.empty(0)
        self.visibility_count = torch.empty(0, dtype=torch.long)
        self.gaussian_age = torch.empty(0, dtype=torch.long)
        self.importance_accum = torch.empty(0)
        self.projected_area_accum = torch.empty(0)
        self.geometry_aware = False
        self._last_densify_counts = {"cloned": 0, "split": 0, "pruned": 0}
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        legacy = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        # Disabled mode emits the exact historic tuple and allocates no stats.
        if not self.geometry_aware:
            return legacy
        # Appending preserves the ability to load every historic 12-tuple.
        return legacy + ((self.residual_accum, self.edge_accum, self.visibility_count,
                          self.gaussian_age, self.importance_accum,
                          self.projected_area_accum),)
    
    def restore(self, model_args, training_args):
        legacy_args = model_args[:12]
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = legacy_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if len(model_args) > 12:
            buffers = model_args[12]
            if len(buffers) == 6 and buffers[0].shape[0] == self.get_xyz.shape[0]:
                (self.residual_accum, self.edge_accum, self.visibility_count,
                 self.gaussian_age, self.importance_accum,
                 self.projected_area_accum) = [item.to(self.get_xyz.device) for item in buffers]
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    @staticmethod
    def _knn_mean_distance(points, k):
        """Bounded-memory KNN distances; never materializes an N-by-N matrix."""
        n = points.shape[0]
        if n < 2:
            return torch.full((n,), 1e-3, device=points.device)
        k = max(1, min(int(k), n - 1))
        if k == 1:
            return torch.sqrt(torch.clamp_min(distCUDA2(points), 1e-8))
        result = torch.empty(n, device=points.device)
        chunk = min(2048, n)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            distances = torch.cdist(points[start:end], points)
            rows = torch.arange(end - start, device=points.device)
            distances[rows, torch.arange(start, end, device=points.device)] = float("inf")
            result[start:end] = distances.topk(k, largest=False).values.mean(dim=1)
        return result.clamp_min(1e-4)

    @staticmethod
    def _quaternion_from_z(normals):
        """Quaternion whose local z axis is aligned to each unit normal."""
        normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-6)
        z = torch.tensor([0.0, 0.0, 1.0], device=normals.device).expand_as(normals)
        cross = torch.cross(z, normals, dim=1)
        dot = (z * normals).sum(1, keepdim=True)
        quat = torch.cat((1.0 + dot, cross), dim=1)
        antiparallel = quat[:, 0].abs() < 1e-6
        quat[antiparallel] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device)
        return torch.nn.functional.normalize(quat, dim=1, eps=1e-6)

    def create_from_dense_prior(self, data, cam_infos, spatial_lr_scale, options):
        """Initialize the standard 3DGS representation from an offline prior."""
        points = torch.as_tensor(data["points"], dtype=torch.float32, device="cuda")
        colors = torch.as_tensor(data["colors"], dtype=torch.float32, device="cuda").clamp(0, 1)
        if points.shape[0] == 0:
            raise ValueError("Dense prior has no valid points after confidence/voxel filtering.")
        self.spatial_lr_scale = spatial_lr_scale
        features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, :, 0] = RGB2SH(colors)
        knn = self._knn_mean_distance(points, options.dense_prior_knn_k)
        scales = torch.log(knn[:, None].repeat(1, 3))
        rotations = torch.zeros((points.shape[0], 4), device="cuda")
        rotations[:, 0] = 1.0
        if options.initialize_rotation_from_normal and data.get("normals") is not None:
            rotations = self._quaternion_from_z(torch.as_tensor(data["normals"], dtype=torch.float32, device="cuda"))
            # Make local z the minor axis, matching the rendered-normal rule.
            scales[:, 2] = scales[:, 2] + np.log(0.8)
        opacity = 0.1 * torch.ones((points.shape[0], 1), device="cuda")
        if options.initialize_opacity_from_confidence and data.get("confidence") is not None:
            confidence = torch.as_tensor(data["confidence"], dtype=torch.float32, device="cuda").clamp(0, 1)
            opacity = 0.01 + 0.49 * confidence[:, None]
        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, :1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(self.inverse_opacity_activation(opacity).requires_grad_(True))
        self.max_radii2D = torch.zeros(points.shape[0], device="cuda")
        self.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cam_infos)}
        self.pretrained_exposures = None
        self._exposure = nn.Parameter(torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1).requires_grad_(True))
        print(f"BTS-GeoGS dense initialization: {points.shape[0]} points")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.geometry_aware = getattr(training_args, "geometry_aware", False)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._reset_geometry_buffers()

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def _reset_geometry_buffers(self):
        """Initialize side statistics without adding optimizer parameters."""
        n, device = self.get_xyz.shape[0], self.get_xyz.device
        if not self.geometry_aware:
            self.residual_accum = torch.empty(0, device=device)
            self.edge_accum = torch.empty(0, device=device)
            self.visibility_count = torch.empty(0, device=device, dtype=torch.long)
            self.gaussian_age = torch.empty(0, device=device, dtype=torch.long)
            self.importance_accum = torch.empty(0, device=device)
            self.projected_area_accum = torch.empty(0, device=device)
            return
        self.residual_accum = torch.zeros((n, 1), device=device)
        self.edge_accum = torch.zeros((n, 1), device=device)
        self.visibility_count = torch.zeros((n, 1), device=device, dtype=torch.long)
        self.gaussian_age = torch.zeros((n, 1), device=device, dtype=torch.long)
        self.importance_accum = torch.zeros((n, 1), device=device)
        self.projected_area_accum = torch.zeros((n, 1), device=device)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        for name in ("residual_accum", "edge_accum", "visibility_count", "gaussian_age",
                     "importance_accum", "projected_area_accum"):
            value = getattr(self, name)
            if value.numel() == valid_points_mask.shape[0]:
                setattr(self, name, value[valid_points_mask])
        if getattr(self, "tmp_radii", None) is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        if self.geometry_aware:
            count = new_xyz.shape[0]
            def extend(name):
                value = getattr(self, name)
                return torch.cat((value, torch.zeros((count, value.shape[1]), device=value.device, dtype=value.dtype)), dim=0)
            self.xyz_gradient_accum = extend("xyz_gradient_accum")
            self.denom = extend("denom")
            self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros(count, device=self.max_radii2D.device)))
            for name in ("residual_accum", "edge_accum", "visibility_count", "gaussian_age",
                         "importance_accum", "projected_area_accum"):
                setattr(self, name, extend(name))
        else:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, selected_pts_mask=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return 0

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return int(selected_pts_mask.sum().item())

    def densify_and_clone(self, grads, grad_threshold, scene_extent, selected_pts_mask=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return 0
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        return int(selected_pts_mask.sum().item())

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is None:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def advance_gaussian_age(self):
        if self.geometry_aware and self.gaussian_age.numel():
            self.gaussian_age.add_(1)

    def accumulate_geometry_stats(self, camera, visibility_filter, residual_map, edge_map, radii):
        """Accumulate per-Gaussian residual/edge statistics from projected centers."""
        if not self.geometry_aware or self.get_xyz.numel() == 0:
            return
        visible = (radii > 0).reshape(-1)
        if not visible.any():
            return
        homogeneous = torch.cat((self.get_xyz.detach(), torch.ones_like(self.get_xyz[:, :1])), dim=1)
        clip = homogeneous @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-6)
        x = ((ndc[:, 0] + 1.0) * 0.5 * (camera.image_width - 1)).round().long().clamp(0, camera.image_width - 1)
        y = ((ndc[:, 1] + 1.0) * 0.5 * (camera.image_height - 1)).round().long().clamp(0, camera.image_height - 1)
        residual = torch.nan_to_num(residual_map.detach())[y, x].unsqueeze(1)
        edge = torch.nan_to_num(edge_map.detach().squeeze(0))[y, x].unsqueeze(1)
        visible_2d = visible.unsqueeze(1)
        self.residual_accum[visible_2d] += residual[visible_2d]
        self.edge_accum[visible_2d] += edge[visible_2d]
        self.visibility_count[visible_2d] += 1
        area = radii.detach().float().square().unsqueeze(1)
        self.projected_area_accum[visible_2d] += area[visible_2d]
        self.importance_accum[visible_2d] += self.get_opacity.detach()[visible_2d]

    @staticmethod
    def _limit_mask(mask, score, limit):
        if limit <= 0:
            return torch.zeros_like(mask)
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if indices.numel() <= limit:
            return mask
        top = indices[torch.topk(score[indices], k=limit).indices]
        result = torch.zeros_like(mask)
        result[top] = True
        return result

    def densify_and_prune_geometry(self, max_grad, min_opacity, extent, max_screen_size, radii, opt):
        """Bounded residual/edge-aware densification; baseline uses its old path."""
        self.tmp_radii = radii
        denom = self.denom.clamp_min(1.0)
        gradient = torch.nan_to_num((self.xyz_gradient_accum / denom).squeeze(1))
        residual = (self.residual_accum / self.visibility_count.clamp_min(1)).squeeze(1)
        edge = (self.edge_accum / self.visibility_count.clamp_min(1)).squeeze(1)
        score = float(opt.densification_gradient_weight) * gradient
        if opt.densification_residual_aware:
            score = score + float(opt.densification_residual_weight) * residual * float(max_grad)
        if opt.densification_edge_aware:
            score = score + float(opt.densification_edge_weight) * edge * float(max_grad)
        eligible = (score >= max_grad) & (self.visibility_count.squeeze(1) >= opt.importance_pruning_min_visibility_count)
        eligible &= self.gaussian_age.squeeze(1) >= int(opt.min_gaussian_age)
        small = self.get_scaling.max(dim=1).values <= self.percent_dense * extent
        clone_mask, split_mask = eligible & small, eligible & ~small
        budget = max(0, int(opt.max_gaussians) - self.get_xyz.shape[0])
        clone_mask = self._limit_mask(clone_mask, score, budget)
        budget -= int(clone_mask.sum().item())
        # A split temporarily appends two children before pruning its parent.
        # Reserve both slots so MAX_GAUSSIANS is respected even mid-operation.
        split_mask = self._limit_mask(split_mask, score, budget // 2)
        cloned = self.densify_and_clone(gradient.unsqueeze(1), max_grad, extent, clone_mask)
        if cloned:
            gradient = torch.cat((gradient, torch.zeros(cloned, device=gradient.device)))
            split_mask = torch.cat((split_mask, torch.zeros(cloned, device=split_mask.device, dtype=torch.bool)))
        split = self.densify_and_split(gradient.unsqueeze(1), max_grad, extent, 2, split_mask)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            prune_mask |= (self.max_radii2D > max_screen_size)
            prune_mask |= (self.get_scaling.max(dim=1).values > 0.1 * extent)
        pruned = int(prune_mask.sum().item())
        if pruned and pruned < self.get_xyz.shape[0]:
            self.prune_points(prune_mask)
        self.tmp_radii = None
        self._last_densify_counts = {"cloned": cloned, "split": split, "pruned": pruned}

    def prune_low_importance(self, opt):
        if self.get_xyz.shape[0] <= 1:
            return 0
        visibility = self.visibility_count.float().clamp_min(1.0)
        importance = (self.importance_accum / visibility) * torch.log1p(self.visibility_count.float()) * (self.projected_area_accum / visibility)
        self.last_mean_importance = importance.mean().item()
        low = importance.squeeze(1) < float(opt.importance_pruning_threshold)
        weak = (self.get_opacity.squeeze(1) < float(opt.importance_pruning_min_opacity)) | (self.visibility_count.squeeze(1) < int(opt.importance_pruning_min_visibility_count))
        mask = low & weak & (self.gaussian_age.squeeze(1) > int(opt.min_gaussian_age))
        if mask.any() and (~mask).any():
            count = int(mask.sum().item())
            self.prune_points(mask)
            self._last_densify_counts["pruned"] += count
            return count
        return 0
