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
import warnings
from types import SimpleNamespace
import torch.nn.functional as F
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.training_schedules import get_lr_multipliers
from utils.exposure_utils import PerViewExposure, TemporalExposureSpline, frame_time_from_name
from utils.sky_utils import DirectionalSHBackground
from utils.densification_utils import (
    accumulate_visible_statistics,
    limit_mask,
    percentile_mask,
    robust_normalize_score,
    conservative_opacity_prune_mask,
    split_opacity_conserving,
)
from utils.footprint_sampling import project_gaussian_ellipse, sample_footprint_statistics
from utils.structure_aligned_split import classify_gaussian_shape, structure_aligned_split
from utils.spatial_densification import spatially_balanced_topk, tile_error_energy
from utils.persistent_densification import (
    bitset_popcount,
    compute_persistent_scores,
    recent_hit_count,
    update_persistent_window,
    update_view_support,
)


_PERSISTENT_BUFFER_NAMES = (
    "persistent_score_ema", "persistent_hit_ema", "window_hit_count",
    "recent_window_mask", "unique_view_support", "view_direction_support",
    "depth_consistent_support", "sky_support", "low_parallax_support",
    "gradient_burstiness", "last_densified_iteration", "persistent_edge_ema",
)

_INTERVAL_BUFFER_NAMES = (
    "xyz_gradient_abs_accum", "xyz_gradient_sq_accum", "residual_accum",
    "residual_denom", "edge_accum", "visibility_count", "gaussian_age",
    "importance_accum", "projected_area_accum", "footprint_residual_accum",
    "footprint_edge_accum", "footprint_denom",
)


def _dist_cuda2(points):
    """Load simple-knn only for point-cloud initialization paths that need it."""
    try:
        from simple_knn._C import distCUDA2
    except ImportError as error:
        raise ImportError(
            "The simple-knn CUDA extension is required to initialize Gaussians from a point cloud. "
            "Rebuild submodules/simple-knn in the active environment. Loading and rendering an "
            "existing Stage 1 checkpoint does not require this extension."
        ) from error
    return distCUDA2(points)

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
        self.xyz_gradient_abs_accum = torch.empty(0)
        self.xyz_gradient_sq_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.residual_accum = torch.empty(0)
        self.residual_denom = torch.empty(0)
        self.edge_accum = torch.empty(0)
        self.visibility_count = torch.empty(0, dtype=torch.long)
        self.gaussian_age = torch.empty(0, dtype=torch.long)
        self.importance_accum = torch.empty(0)
        self.projected_area_accum = torch.empty(0)
        self.footprint_residual_accum = torch.empty(0)
        self.footprint_edge_accum = torch.empty(0)
        self.footprint_denom = torch.empty(0)
        for name in _PERSISTENT_BUFFER_NAMES:
            setattr(self, name, torch.empty(0))
        self.geometry_aware = False
        self.densification_stats_enabled = False
        self.exposure_enabled = False
        self.exposure_model = None
        self.exposure_spline = None
        self.exposure_mapping = {}
        self.pretrained_exposures = None
        self.exposure_inference_options = None
        self.last_exposure_diagnostics = {}
        self.background_model = None
        self.background_optimizer = None
        self._approx_absgrad_warning_printed = False
        self._last_densification_metrics = {}
        self._last_densify_counts = {"cloned": 0, "split": 0, "pruned": 0}
        self._density_control_metrics = {}
        self._structure_split_metrics = {}
        self._last_spatial_context = None
        self._last_spatial_selection = None
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
        payload = {}
        if self.densification_stats_enabled:
            payload["densification_buffers"] = (
                self.xyz_gradient_abs_accum, self.xyz_gradient_sq_accum,
                self.residual_accum, self.residual_denom, self.edge_accum,
                self.visibility_count, self.gaussian_age, self.importance_accum,
                self.projected_area_accum,
            )
            payload["persistent_buffers"] = {
                name: getattr(self, name) for name in _PERSISTENT_BUFFER_NAMES
            }
        if self.exposure_enabled and self.exposure_model is not None:
            payload["exposure"] = {
                "raw_gain": self.exposure_model.raw_gain.detach(),
                "raw_bias": self.exposure_model.raw_bias.detach(),
                "min_gain": self.exposure_model.min_gain,
                "max_gain": self.exposure_model.max_gain,
                "max_bias": self.exposure_model.max_bias,
                "mapping": dict(self.exposure_mapping),
                "camera_positions": self.exposure_model.camera_positions.detach(),
                "camera_directions": self.exposure_model.camera_directions.detach(),
                "camera_focals": self.exposure_model.camera_focals.detach(),
                "camera_times": self.exposure_model.camera_times.detach(),
                "optimizer": self.exposure_optimizer.state_dict(),
            }
            if self.exposure_spline is not None:
                payload["exposure"]["spline_state"] = self.exposure_spline.state_dict()
        if self.background_model is not None:
            payload["background"] = {
                "degree": self.background_model.degree,
                "state_dict": self.background_model.state_dict(),
                "optimizer": (self.background_optimizer.state_dict()
                              if self.background_optimizer is not None else None),
            }
        # Disabled mode emits the exact historic tuple.
        if not payload:
            return legacy
        return legacy + (payload,)
    
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
            payload = model_args[12]
            if isinstance(payload, dict):
                buffers = payload.get("densification_buffers")
                if buffers is not None and len(buffers) == 9 and buffers[0].shape[0] == self.get_xyz.shape[0]:
                    (self.xyz_gradient_abs_accum, self.xyz_gradient_sq_accum, self.residual_accum,
                     self.residual_denom, self.edge_accum, self.visibility_count, self.gaussian_age,
                     self.importance_accum, self.projected_area_accum) = [item.to(self.get_xyz.device) for item in buffers]
                persistent = payload.get("persistent_buffers", {})
                for name in _PERSISTENT_BUFFER_NAMES:
                    value = persistent.get(name)
                    if value is not None and value.shape[0] == self.get_xyz.shape[0]:
                        setattr(self, name, value.to(self.get_xyz.device))
                exposure = payload.get("exposure")
                if exposure is not None:
                    self._restore_exposure_payload(exposure)
                background = payload.get("background")
                if background is not None and self.background_model is not None:
                    self.background_model.load_state_dict(background["state_dict"])
                    if self.background_optimizer is not None and background.get("optimizer") is not None:
                        self.background_optimizer.load_state_dict(background["optimizer"])
            else:
                # GeoGS-v1/v2 appended tuple compatibility.
                buffers = payload
                if len(buffers) in (6, 7) and buffers[0].shape[0] == self.get_xyz.shape[0]:
                    if len(buffers) == 6:
                        (self.residual_accum, self.edge_accum, self.visibility_count, self.gaussian_age,
                         self.importance_accum, self.projected_area_accum) = [item.to(self.get_xyz.device) for item in buffers]
                    else:
                        (self.xyz_gradient_abs_accum, self.residual_accum, self.edge_accum, self.visibility_count,
                         self.gaussian_age, self.importance_accum, self.projected_area_accum) = [item.to(self.get_xyz.device) for item in buffers]
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
        if self.exposure_spline is not None:
            gain, bias = self.exposure_spline.gains_biases()
            matrices = gain.new_zeros((gain.shape[0], 3, 4))
            matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2] = gain.unbind(dim=1)
            matrices[:, :, 3] = bias
            return matrices
        return self.exposure_model.as_matrices() if self.exposure_model is not None else self.get_xyz.new_empty((0, 3, 4))

    def get_exposure_from_name(self, image_name):
        if self.exposure_model is None or image_name not in self.exposure_mapping:
            matrix = self.get_xyz.new_zeros((3, 4))
            matrix[:, :3] = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
            return matrix
        return self.get_exposure[self.exposure_mapping[image_name]]

    def setup_exposure(self, cam_infos, options=None):
        min_gain = float(getattr(options, "exposure_min_gain", 0.75))
        max_gain = float(getattr(options, "exposure_max_gain", 1.25))
        max_bias = float(getattr(options, "exposure_max_bias", 0.10))
        device = self.get_xyz.device if self.get_xyz.is_cuda else torch.device("cuda")
        self.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cam_infos)}
        self.exposure_inference_options = options
        self.exposure_model = PerViewExposure(len(cam_infos), min_gain, max_gain, max_bias).to(device)
        positions, directions, focals, times = [], [], [], []
        for cam in cam_infos:
            rotation = np.asarray(cam.R, dtype=np.float32)
            translation = np.asarray(cam.T, dtype=np.float32)
            positions.append(-rotation @ translation)
            directions.append(rotation[:, 2])
            focals.append(0.5 * (float(getattr(cam, "fx", 0.0)) + float(getattr(cam, "fy", 0.0))))
            times.append(frame_time_from_name(cam.image_name))
        if positions:
            valid_focals = focals if all(value > 0 for value in focals) else None
            valid_times = times if all(value is not None for value in times) else None
            self.exposure_model.set_camera_poses(np.stack(positions), np.stack(directions), valid_focals, valid_times)
        self.exposure_spline = None
        if bool(getattr(options, "exposure_spline_enabled", False)):
            self.exposure_spline = TemporalExposureSpline(
                times, num_knots=int(getattr(options, "exposure_spline_num_knots", 12)),
                degree=int(getattr(options, "exposure_spline_degree", 3)),
                min_gain=min_gain, max_gain=max_gain, max_bias=max_bias,
                per_view_residual=bool(getattr(options, "exposure_per_view_residual_enabled", True)),
            ).to(device)

    def setup_background(self, options=None, initial_color=(0.0, 0.0, 0.0)):
        """Create the optional low-order sky model before optimizer setup."""
        mode = str(getattr(options, "sky_background_mode", "constant")) if options is not None else "constant"
        enabled = bool(getattr(options, "sky_enabled", False)) if options is not None else False
        if enabled and mode == "directional_sh":
            degree = int(getattr(options, "sky_background_degree", 2))
            device = self.get_xyz.device if self.get_xyz.is_cuda else torch.device("cuda")
            self.background_model = DirectionalSHBackground(degree, initial_color).to(device)
        else:
            self.background_model = None

    def apply_exposure(self, image, camera, mode="training", k=4):
        if self.exposure_model is None:
            return image
        if mode in (None, "training", "learned") and camera.image_name in self.exposure_mapping:
            if self.exposure_spline is not None:
                return self.exposure_spline(image, self.exposure_mapping[camera.image_name])
            return self.exposure_model(image, self.exposure_mapping[camera.image_name])
        options = self.exposure_inference_options
        if self.exposure_spline is not None and mode == "temporal_spline":
            inferred = self.exposure_spline.infer_time(getattr(camera, "frame_time", None))
            if inferred is not None:
                gain, bias = inferred
                diagnostics = {"mode": "temporal_spline", "confidence": 1.0,
                               "indices": [], "weights": []}
                corrected = gain[:, None, None] * image + bias[:, None, None]
                self.last_exposure_diagnostics = diagnostics
                return corrected
            mode = "pose_confidence_blend"
        gain, bias, diagnostics = self.exposure_model.infer_gain_bias(
            camera.camera_center, np.asarray(camera.R, dtype=np.float32)[:, 2], mode or "identity",
            k=int(getattr(options, "exposure_test_k", k)), focal=0.5 * (camera.fx + camera.fy),
            time=getattr(camera, "frame_time", None),
            position_weight=float(getattr(options, "exposure_position_weight", 1.0)),
            angle_weight=float(getattr(options, "exposure_angle_weight", 0.5)),
            temporal_weight=float(getattr(options, "exposure_temporal_weight", 0.0)),
            focal_weight=float(getattr(options, "exposure_focal_weight", 0.0)),
            distance_temperature=float(getattr(options, "exposure_distance_temperature", 0.10)),
            confidence_temperature=float(getattr(options, "exposure_confidence_temperature", 0.08)),
            min_confidence=float(getattr(options, "exposure_min_confidence", 0.0)),
            max_gain_delta=float(getattr(options, "exposure_max_gain_delta_at_test", 0.20)),
            max_bias=float(getattr(options, "exposure_max_bias_at_test", 0.08)),
            return_diagnostics=True)
        if self.exposure_spline is not None and diagnostics.get("indices"):
            spline_gains, spline_biases = self.exposure_spline.gains_biases()
            indices = torch.as_tensor(diagnostics["indices"], device=spline_gains.device, dtype=torch.long)
            weights = torch.as_tensor(diagnostics["weights"], device=spline_gains.device, dtype=spline_gains.dtype)
            gain = (weights[:, None] * spline_gains[indices]).sum(dim=0)
            bias = (weights[:, None] * spline_biases[indices]).sum(dim=0)
            confidence_value = float(diagnostics.get("confidence", 1.0))
            gain = 1.0 + confidence_value * (gain - 1.0)
            bias = confidence_value * bias
        self.last_exposure_diagnostics = diagnostics
        corrected = gain[:, None, None] * image + bias[:, None, None]
        self.last_exposure_diagnostics["out_of_range_fraction"] = float(
            self.exposure_out_of_range_fraction(corrected).detach().item())
        return corrected

    def exposure_regularization(self, gain_weight, bias_weight, zero_mean_weight=0.0):
        if self.exposure_model is None:
            return self.get_xyz.sum() * 0.0
        if self.exposure_spline is not None:
            return self.exposure_spline.regularization_loss(
                float(getattr(self.training_args, "exposure_spline_smoothness_weight", 0.01)),
                float(getattr(self.training_args, "exposure_per_view_residual_weight", 0.01)))
        return self.exposure_model.regularization_loss(gain_weight, bias_weight, zero_mean_weight)

    def exposure_out_of_range_fraction(self, image):
        return ((image < 0.0) | (image > 1.0)).float().mean()

    def _restore_exposure_payload(self, payload):
        if self.exposure_model is None:
            return
        raw_gain = payload.get("raw_gain")
        raw_bias = payload.get("raw_bias")
        if raw_gain is not None and raw_gain.shape == self.exposure_model.raw_gain.shape:
            with torch.no_grad():
                self.exposure_model.raw_gain.copy_(raw_gain.to(self.exposure_model.raw_gain.device))
                self.exposure_model.raw_bias.copy_(raw_bias.to(self.exposure_model.raw_bias.device))
        positions, directions = payload.get("camera_positions"), payload.get("camera_directions")
        if positions is not None and positions.shape == self.exposure_model.camera_positions.shape:
            self.exposure_model.set_camera_poses(
                positions, directions,
                payload.get("camera_focals", self.exposure_model.camera_focals),
                payload.get("camera_times", self.exposure_model.camera_times))
        optimizer_state = payload.get("optimizer")
        if optimizer_state is not None:
            try:
                self.exposure_optimizer.load_state_dict(optimizer_state)
            except (ValueError, KeyError) as error:
                warnings.warn(f"Exposure optimizer state was incompatible and was reinitialized: {error}")
        spline_state = payload.get("spline_state")
        if spline_state is not None and self.exposure_spline is not None:
            try:
                self.exposure_spline.load_state_dict(spline_state)
            except (ValueError, RuntimeError, KeyError) as error:
                warnings.warn(f"Exposure spline state was incompatible and was reinitialized: {error}")
        elif spline_state is not None:
            warnings.warn("Checkpoint contains an exposure spline but the current config disables it.")

    def save_exposure_json(self, path):
        if self.exposure_model is None:
            return
        if self.exposure_spline is not None:
            spline_gain, spline_bias = self.exposure_spline.gains_biases()
            gains, biases = spline_gain.detach().cpu(), spline_bias.detach().cpu()
            matrices = self.get_exposure.detach().cpu()
        else:
            gains = self.exposure_model.gains().detach().cpu()
            biases = self.exposure_model.biases().detach().cpu()
            matrices = self.exposure_model.as_matrices().detach().cpu()
        inverse_mapping = {index: name for name, index in self.exposure_mapping.items()}
        views = {}
        for index in range(self.exposure_model.num_views):
            views[inverse_mapping[index]] = {
                "gain": gains[index].tolist(), "bias": biases[index].tolist(),
                "matrix": matrices[index].tolist(),
                "position": self.exposure_model.camera_positions[index].detach().cpu().tolist(),
                "view_direction": self.exposure_model.camera_directions[index].detach().cpu().tolist(),
                "focal": (float(self.exposure_model.camera_focals[index].item())
                          if self.exposure_model.camera_focals.shape[0] == self.exposure_model.num_views else None),
                "frame_time": (float(self.exposure_model.camera_times[index].item())
                              if self.exposure_model.camera_times.shape[0] == self.exposure_model.num_views else None),
            }
        inference_names = (
            "exposure_test_k", "exposure_position_weight", "exposure_angle_weight",
            "exposure_temporal_weight", "exposure_focal_weight", "exposure_distance_temperature",
            "exposure_confidence_temperature", "exposure_min_confidence",
            "exposure_max_gain_delta_at_test", "exposure_max_bias_at_test",
        )
        inference = {name: getattr(self.exposure_inference_options, name)
                     for name in inference_names
                     if self.exposure_inference_options is not None
                     and hasattr(self.exposure_inference_options, name)}
        state = {"version": 4, "mode": "diagonal_gain_bias",
                 "min_gain": self.exposure_model.min_gain, "max_gain": self.exposure_model.max_gain,
                 "max_bias": self.exposure_model.max_bias, "views": views, "inference": inference,
                 "legacy_matrices": {name: views[name]["matrix"] for name in views}}
        if self.exposure_spline is not None:
            state["version"] = 5
            state["mode"] = "temporal_spline"
            state["spline"] = {
                "raw_gain_knots": self.exposure_spline.raw_gain_knots.detach().cpu().tolist(),
                "raw_bias_knots": self.exposure_spline.raw_bias_knots.detach().cpu().tolist(),
                "raw_gain_residual": self.exposure_spline.raw_gain_residual.detach().cpu().tolist(),
                "raw_bias_residual": self.exposure_spline.raw_bias_residual.detach().cpu().tolist(),
            }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def load_exposure_json(self, path):
        if self.exposure_model is None or not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        views = state.get("views") if isinstance(state, dict) else None
        if isinstance(state, dict) and state.get("inference"):
            self.exposure_inference_options = SimpleNamespace(**state["inference"])
        if views is None:
            views = state  # Historic image_name -> 3x4 matrix mapping.
        gain = self.exposure_model.gains().detach().clone()
        bias = self.exposure_model.biases().detach().clone()
        for name, index in self.exposure_mapping.items():
            if name not in views:
                continue
            value = views[name]
            if isinstance(value, dict):
                gain[index] = torch.as_tensor(value["gain"], device=gain.device)
                bias[index] = torch.as_tensor(value["bias"], device=bias.device)
            else:
                matrix = torch.as_tensor(value, dtype=gain.dtype, device=gain.device)
                gain[index] = torch.diagonal(matrix[:, :3])
                bias[index] = matrix[:, 3]
        self.exposure_model.load_gain_bias(gain, bias)
        spline = state.get("spline") if isinstance(state, dict) else None
        if spline is not None and self.exposure_spline is not None:
            with torch.no_grad():
                for name in ("raw_gain_knots", "raw_bias_knots",
                             "raw_gain_residual", "raw_bias_residual"):
                    target = getattr(self.exposure_spline, name)
                    value = torch.as_tensor(spline[name], device=target.device, dtype=target.dtype)
                    if value.shape != target.shape:
                        raise ValueError(f"Exposure spline JSON shape mismatch for {name}")
                    target.copy_(value)
        return True

    def save_background(self, path):
        if self.background_model is not None:
            torch.save({"degree": self.background_model.degree,
                        "state_dict": self.background_model.state_dict()}, path)

    def load_background(self, path):
        if not os.path.exists(path):
            return False
        device = self.get_xyz.device if self.get_xyz.is_cuda else torch.device("cuda")
        state = torch.load(path, map_location=device, weights_only=True)
        degree = int(state["degree"])
        if self.background_model is None or self.background_model.degree != degree:
            self.background_model = DirectionalSHBackground(degree).to(device)
        self.background_model.load_state_dict(state["state_dict"])
        return True
    
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

        dist2 = torch.clamp_min(_dist_cuda2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
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

    @staticmethod
    def _knn_mean_distance(points, k):
        """Bounded-memory KNN distances; never materializes an N-by-N matrix."""
        n = points.shape[0]
        if n < 2:
            return torch.full((n,), 1e-3, device=points.device)
        k = max(1, min(int(k), n - 1))
        if k == 1:
            return torch.sqrt(torch.clamp_min(_dist_cuda2(points), 1e-8))
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
        print(f"BTS-GeoGS dense initialization: {points.shape[0]} points")

    def training_setup(self, training_args):
        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        if getattr(training_args, "initialization_cap_to_max_gaussians", False):
            self._cap_initial_gaussians(int(training_args.max_gaussians))
        self.geometry_aware = getattr(training_args, "geometry_aware", False)
        self.densification_stats_enabled = (
            self.geometry_aware or getattr(training_args, "densification_method", "original") != "original"
            or getattr(training_args, "densification_residual_aware", False)
            or getattr(training_args, "densification_edge_aware", False)
        )
        self.exposure_enabled = bool(getattr(training_args, "exposure_compensation", False))
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

        if self.exposure_model is None:
            raise RuntimeError("Scene must initialize the per-view exposure module before training_setup().")
        exposure_parameters = (self.exposure_spline.parameters()
                               if self.exposure_spline is not None else self.exposure_model.parameters())
        self.exposure_optimizer = torch.optim.Adam(exposure_parameters, lr=0.0)
        self.background_optimizer = None
        if self.background_model is not None:
            self.background_optimizer = torch.optim.Adam(
                self.background_model.parameters(),
                lr=float(getattr(training_args, "sky_background_lr", 1e-3)))

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def _cap_initial_gaussians(self, max_points):
        """Deterministically subsample an oversized initialization before Adam state exists."""

        count = int(self.get_xyz.shape[0])
        max_points = int(max_points)
        if max_points <= 0 or count <= max_points:
            return 0
        indices = torch.linspace(
            0, count - 1, steps=max_points, device=self.get_xyz.device).round().long()
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
            value = getattr(self, name).detach()[indices].contiguous()
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))
        self.max_radii2D = self.max_radii2D.detach()[indices].contiguous()
        print(
            f"BTS-GeoGS initialization cap: {count} -> {max_points} Gaussians",
            flush=True)
        return count - max_points

    def _reset_geometry_buffers(self):
        """Initialize side statistics without adding optimizer parameters."""
        n, device = self.get_xyz.shape[0], self.get_xyz.device
        if not self.densification_stats_enabled:
            self.residual_accum = torch.empty(0, device=device)
            self.residual_denom = torch.empty(0, device=device)
            self.xyz_gradient_abs_accum = torch.empty(0, device=device)
            self.xyz_gradient_sq_accum = torch.empty(0, device=device)
            self.edge_accum = torch.empty(0, device=device)
            self.visibility_count = torch.empty(0, device=device, dtype=torch.long)
            self.gaussian_age = torch.empty(0, device=device, dtype=torch.long)
            self.importance_accum = torch.empty(0, device=device)
            self.projected_area_accum = torch.empty(0, device=device)
            self.footprint_residual_accum = torch.empty(0, device=device)
            self.footprint_edge_accum = torch.empty(0, device=device)
            self.footprint_denom = torch.empty(0, device=device)
            for name in _PERSISTENT_BUFFER_NAMES:
                dtype = torch.int64 if name in {"window_hit_count", "recent_window_mask",
                                                "unique_view_support", "view_direction_support",
                                                "last_densified_iteration"} else torch.float32
                setattr(self, name, torch.empty(0, device=device, dtype=dtype))
            return
        self.residual_accum = torch.zeros((n, 1), device=device)
        self.residual_denom = torch.zeros((n, 1), device=device)
        self.xyz_gradient_abs_accum = torch.zeros((n, 1), device=device)
        self.xyz_gradient_sq_accum = torch.zeros((n, 1), device=device)
        self.edge_accum = torch.zeros((n, 1), device=device)
        self.visibility_count = torch.zeros((n, 1), device=device, dtype=torch.long)
        self.gaussian_age = torch.zeros((n, 1), device=device, dtype=torch.long)
        self.importance_accum = torch.zeros((n, 1), device=device)
        self.projected_area_accum = torch.zeros((n, 1), device=device)
        self.footprint_residual_accum = torch.zeros((n, 1), device=device)
        self.footprint_edge_accum = torch.zeros((n, 1), device=device)
        self.footprint_denom = torch.zeros((n, 1), device=device)
        for name in _PERSISTENT_BUFFER_NAMES:
            dtype = torch.int64 if name in {"window_hit_count", "recent_window_mask",
                                            "unique_view_support", "view_direction_support",
                                            "last_densified_iteration"} else torch.float32
            setattr(self, name, torch.zeros((n, 1), device=device, dtype=dtype))

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        exposure_active = (self.exposure_enabled and
                           int(getattr(self.training_args, "exposure_start_iter", 0)) <= iteration <=
                           min(int(getattr(self.training_args, "exposure_end_iter", self.training_args.iterations)),
                               int(getattr(self.training_args, "exposure_freeze_iter", self.training_args.iterations))))
        for param_group in self.exposure_optimizer.param_groups:
            param_group['lr'] = (self.exposure_scheduler_args(iteration) *
                                 get_lr_multipliers(iteration, self.training_args)["exposure"]
                                 if exposure_active else 0.0)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration) * get_lr_multipliers(iteration, self.training_args)["xyz"]
                param_group['lr'] = lr
            elif param_group["name"] == "f_dc" or param_group["name"] == "f_rest":
                param_group['lr'] = self.training_args.feature_lr * get_lr_multipliers(iteration, self.training_args)["features"] * (1.0 if param_group["name"] == "f_dc" else 1.0 / 20.0)
            elif param_group["name"] == "opacity":
                param_group['lr'] = self.training_args.opacity_lr * get_lr_multipliers(iteration, self.training_args)["opacity"]
            elif param_group["name"] == "scaling":
                param_group['lr'] = self.training_args.scaling_lr * get_lr_multipliers(iteration, self.training_args)["scaling"]
            elif param_group["name"] == "rotation":
                param_group['lr'] = self.training_args.rotation_lr * get_lr_multipliers(iteration, self.training_args)["rotation"]
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
        for name in _INTERVAL_BUFFER_NAMES + _PERSISTENT_BUFFER_NAMES:
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
        if self.densification_stats_enabled:
            count = new_xyz.shape[0]
            def extend(name):
                value = getattr(self, name)
                return torch.cat((value, torch.zeros((count, value.shape[1]), device=value.device, dtype=value.dtype)), dim=0)
            self.xyz_gradient_accum = extend("xyz_gradient_accum")
            self.denom = extend("denom")
            self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros(count, device=self.max_radii2D.device)))
            for name in _INTERVAL_BUFFER_NAMES + _PERSISTENT_BUFFER_NAMES:
                setattr(self, name, extend(name))
        else:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _record_density_control(self, prefix, parent_raw, child_raw, num_outputs):
        parent_alpha = torch.sigmoid(parent_raw.detach())
        child_alpha = torch.sigmoid(child_raw.detach())
        self._density_control_metrics.update({
            f"{prefix}_mean_parent_alpha": float(parent_alpha.mean().item()),
            f"{prefix}_mean_child_alpha": float(child_alpha.mean().item()),
            f"{prefix}_alpha_mass_before": float(parent_alpha.sum().item()),
            f"{prefix}_alpha_mass_after": float(child_alpha.sum().item()),
            f"{prefix}_near_saturation_fraction": float((child_alpha > 0.99).float().mean().item()),
            f"{prefix}_num_outputs": int(num_outputs),
        })

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, selected_pts_mask=None,
                          options=None):
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

        options = self.training_args if options is None else options
        parent_xyz = self.get_xyz[selected_pts_mask]
        parent_scaling = self.get_scaling[selected_pts_mask]
        parent_rotation = self._rotation[selected_pts_mask]
        parent_opacity = self._opacity[selected_pts_mask]
        if getattr(options, "structure_aligned_split_enabled", False):
            result = structure_aligned_split(
                parent_xyz, parent_scaling, parent_rotation, parent_opacity,
                error_direction=None, cfg=options, num_outputs=N)
            new_xyz = result.xyz
            new_scaling = self.scaling_inverse_activation(result.scaling)
            new_rotation = result.rotation
            new_opacity = result.raw_opacity
            wire, surface, blob = classify_gaussian_shape(
                parent_scaling, float(options.wire_ratio_threshold), float(options.surface_ratio_threshold))
            self._structure_split_metrics = {
                "wire_like_count": int(wire.sum().item()),
                "surface_like_count": int(surface.sum().item()),
                "blob_like_count": int(blob.sum().item()),
                "wire_split_count": int(wire.sum().item()),
                "surface_split_count": int(surface.sum().item()),
                "blob_split_count": int(blob.sum().item()),
                "child_parent_distance_mean": float(result.parent_distance.mean().item()),
            }
        else:
            stds = parent_scaling.repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device=stds.device, dtype=stds.dtype)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(parent_rotation).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + parent_xyz.repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(parent_scaling.repeat(N, 1) / (0.8 * N))
            new_rotation = parent_rotation.repeat(N, 1)
            child_raw = (split_opacity_conserving(parent_opacity, N)
                         if getattr(options, "opacity_conserving_split", False) else parent_opacity)
            new_opacity = child_raw.repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self._record_density_control("split", parent_opacity, new_opacity, N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device=selected_pts_mask.device, dtype=torch.bool)))
        self.prune_points(prune_filter)
        return int(selected_pts_mask.sum().item())

    def densify_and_clone(self, grads, grad_threshold, scene_extent, selected_pts_mask=None,
                          options=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return 0
        
        options = self.training_args if options is None else options
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        parent_opacity = self._opacity[selected_pts_mask].detach().clone()
        if getattr(options, "opacity_conserving_clone", False):
            new_opacities = split_opacity_conserving(parent_opacity, 2)
            with torch.no_grad():
                self._opacity[selected_pts_mask].copy_(new_opacities)
        else:
            new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        combined_opacity = torch.cat((self._opacity[selected_pts_mask], new_opacities), dim=0)
        self._record_density_control("clone", parent_opacity, combined_opacity, 2)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        return int(selected_pts_mask.sum().item())

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent, options=self.training_args)
        self.densify_and_split(grads, max_grad, extent, options=self.training_args)

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
        gradient_xy = viewspace_point_tensor.grad[update_filter, :2]
        self.xyz_gradient_accum[update_filter] += torch.norm(gradient_xy, dim=-1, keepdim=True)
        if self.densification_stats_enabled:
            squared = gradient_xy.square().sum(dim=-1, keepdim=True)
            self.xyz_gradient_sq_accum[update_filter] += squared
            method = getattr(self.training_args, "densification_method", "original")
            abs_gradient = getattr(viewspace_point_tensor, "absgrad", None)
            if abs_gradient is not None:
                abs_xy = abs_gradient[update_filter, :2]
                self.xyz_gradient_abs_accum[update_filter] += torch.norm(abs_xy, dim=-1, keepdim=True)
            elif method == "absolute_gradient_approx":
                self.xyz_gradient_abs_accum[update_filter] += torch.norm(gradient_xy.abs(), dim=-1, keepdim=True)
                if not self._approx_absgrad_warning_printed:
                    print("[BTS-GeoGS] Using approximate post-aggregation absolute gradient. "
                          "This is not full per-pixel AbsGS.")
                    self._approx_absgrad_warning_printed = True
            elif method in ("absolute_gradient", "hybrid", "persistent_multiview_hybrid"):
                raise RuntimeError(
                    "True AbsGS was requested but the installed rasterizer does not expose per-pixel absolute "
                    "mean gradients. Rebuild submodules/diff-gaussian-rasterization or select "
                    "--densification_method absolute_gradient_approx.")
        self.denom[update_filter] += 1

    @staticmethod
    def compute_densification_score(original_grad, absolute_grad=None, residual=None, edge_score=None,
                                    rms_grad=None, cfg=None, valid_mask=None):
        method = getattr(cfg, "densification_method", "original")
        if method == "original":
            return original_grad
        if method in ("absolute_gradient", "absolute_gradient_approx"):
            return absolute_grad if absolute_grad is not None else original_grad
        if method == "rms_gradient":
            return rms_grad if rms_grad is not None else original_grad
        if method == "residual":
            return residual if residual is not None else torch.zeros_like(original_grad)
        valid_mask = torch.ones_like(original_grad, dtype=torch.bool) if valid_mask is None else valid_mask
        score = float(cfg.densification_original_grad_weight) * robust_normalize_score(original_grad, valid_mask)
        if absolute_grad is not None and float(cfg.densification_abs_grad_weight):
            score = score + float(cfg.densification_abs_grad_weight) * robust_normalize_score(absolute_grad, valid_mask)
        if residual is not None and float(cfg.densification_residual_weight):
            score = score + float(cfg.densification_residual_weight) * robust_normalize_score(residual, valid_mask)
        if edge_score is not None and float(cfg.densification_edge_weight):
            score = score + float(cfg.densification_edge_weight) * robust_normalize_score(edge_score, valid_mask)
        return score

    def advance_gaussian_age(self):
        if self.densification_stats_enabled and self.gaussian_age.numel():
            self.gaussian_age.add_(1)

    def accumulate_visibility_only(self, visibility_filter):
        """Track support for gradient-only density control without image sampling.

        AbsGS and RMS-gradient densification only consume projected gradients.
        Projecting every Gaussian again and sampling residual/edge maps adds a
        large O(number-of-Gaussians) Python/Torch pass that those methods do not
        use.  Keep the shared visibility gate while avoiding that overhead.
        """
        if not self.densification_stats_enabled or not self.visibility_count.numel():
            return
        visible = visibility_filter.reshape(-1).bool()
        self.visibility_count[visible] += 1

    def accumulate_geometry_stats(self, camera, visibility_filter, residual_map, edge_map, radii,
                                  depth_confidence_map=None):
        """Accumulate per-Gaussian residual/edge statistics from projected centers."""
        if not self.densification_stats_enabled or self.get_xyz.numel() == 0:
            return
        visible = (radii > 0).reshape(-1)
        if not visible.any():
            return
        homogeneous = torch.cat((self.get_xyz.detach(), torch.ones_like(self.get_xyz[:, :1])), dim=1)
        clip = homogeneous @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-6)
        grid = ndc[:, :2].clamp(-1.0, 1.0).view(1, -1, 1, 2)
        residual_source = torch.nan_to_num(residual_map.detach()).reshape(1, 1, camera.image_height, camera.image_width)
        edge_source = torch.nan_to_num(edge_map.detach()).reshape(1, 1, camera.image_height, camera.image_width)
        residual = F.grid_sample(residual_source, grid, mode="bilinear", align_corners=False).reshape(-1, 1)
        edge = F.grid_sample(edge_source, grid, mode="bilinear", align_corners=False).reshape(-1, 1)
        def sample_optional(value):
            if value is None:
                return torch.zeros_like(residual)
            source = torch.nan_to_num(value.detach()).to(grid.device, non_blocking=True).reshape(
                1, 1, camera.image_height, camera.image_width)
            return F.grid_sample(source, grid, mode="bilinear", align_corners=False).reshape(-1, 1)
        sky = sample_optional(getattr(camera, "sky_mask", None))
        low_parallax = sample_optional(getattr(camera, "low_parallax_mask", None))
        depth_support = sample_optional(depth_confidence_map)
        visible_2d = visible.unsqueeze(1)
        accumulate_visible_statistics(self.residual_accum, self.residual_denom, residual, visible)
        self.edge_accum[visible_2d] += edge[visible_2d]
        self.visibility_count[visible_2d] += 1
        area = radii.detach().float().square().unsqueeze(1)
        self.projected_area_accum[visible_2d] += area[visible_2d]
        self.importance_accum[visible_2d] += self.get_opacity.detach()[visible_2d]
        self.sky_support[visible_2d] += sky[visible_2d]
        self.low_parallax_support[visible_2d] += low_parallax[visible_2d]
        self.depth_consistent_support[visible_2d] += depth_support[visible_2d]
        if getattr(self.training_args, "footprint_sampling_enabled", False):
            footprint_mask = visible & (radii >= float(self.training_args.footprint_min_screen_radius))
            if footprint_mask.any():
                covariance = self.covariance_activation(
                    self.get_scaling[footprint_mask].detach(), 1.0,
                    self._rotation[footprint_mask].detach())
                centers, axis_major, axis_minor = project_gaussian_ellipse(
                    self.get_xyz[footprint_mask].detach(), covariance, camera)
                statistics = sample_footprint_statistics(
                    residual_map.detach(), edge_map.detach(), centers, axis_major, axis_minor,
                    pattern=str(self.training_args.footprint_pattern),
                    mean_weight=float(self.training_args.footprint_mean_weight),
                    p90_weight=float(self.training_args.footprint_p90_weight),
                    max_weight=float(self.training_args.footprint_max_weight))
                selected = torch.nonzero(footprint_mask, as_tuple=False).squeeze(1)
                valid_samples = statistics["valid_sample_count"] > 0
                selected = selected[valid_samples]
                if selected.numel():
                    self.footprint_residual_accum[selected, 0] += statistics["residual_score"][valid_samples]
                    self.footprint_edge_accum[selected, 0] += statistics["edge_score"][valid_samples]
                    self.footprint_denom[selected, 0] += 1
        if getattr(self.training_args, "densification_unique_view_support_enabled", False):
            directions = camera.camera_center[None] - self.get_xyz.detach()
            update_view_support(self.view_direction_support, directions, visible,
                                int(self.training_args.densification_unique_view_bins))
        if getattr(self.training_args, "spatial_budget_enabled", False):
            self._last_spatial_context = {
                "centers": ndc[:, :2].detach(),
                "visible": visible.detach(),
                "tile_energy": tile_error_energy(
                    residual_map.detach(), edge_map.detach(),
                    tile_size=int(self.training_args.spatial_tile_size),
                    edge_weight=float(self.training_args.spatial_tile_edge_weight)),
                "image_width": int(camera.image_width),
                "image_height": int(camera.image_height),
            }

    @staticmethod
    def _limit_mask(mask, score, limit):
        return limit_mask(mask, score, limit)

    def thin_structure_mask(self, opt, iteration=0):
        """Protect stable anisotropic edge Gaussians from opacity-only pruning."""
        empty = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        if (not getattr(opt, "thin_structure_protection", False)
                or iteration > int(getattr(opt, "thin_protection_decay_after_iter", opt.iterations))
                or self.persistent_edge_ema.numel() == 0):
            return empty
        scaling = self.get_scaling.detach().clamp_min(1e-8)
        anisotropy = scaling.max(dim=1).values / scaling.min(dim=1).values
        visibility = self.visibility_count.float().clamp_min(1.0)
        projected_area = (self.projected_area_accum / visibility).squeeze(1)
        return (
            (anisotropy >= float(opt.thin_min_anisotropy))
            & (self.persistent_edge_ema.squeeze(1) >= float(opt.thin_min_edge_support))
            & (self.unique_view_support.squeeze(1) >= int(opt.thin_min_view_bins))
            & (projected_area <= float(opt.thin_max_projected_area))
            & (self.gradient_burstiness.squeeze(1) <= 0.75)
        )

    def densify_and_prune_geometry(self, max_grad, min_opacity, extent, max_screen_size, radii, opt,
                                   iteration=0):
        """Bounded residual/edge-aware densification; baseline uses its old path."""
        self.tmp_radii = radii
        denom = self.denom.clamp_min(1.0)
        gradient = torch.nan_to_num((self.xyz_gradient_accum / denom).squeeze(1))
        absolute_gradient = torch.nan_to_num((self.xyz_gradient_abs_accum / denom).squeeze(1))
        rms_gradient = torch.sqrt(torch.nan_to_num((self.xyz_gradient_sq_accum / denom).squeeze(1)).clamp_min(0.0))
        residual_denom = self.residual_denom.clamp_min(1.0)
        residual = torch.nan_to_num((self.residual_accum / residual_denom).squeeze(1))
        edge = torch.nan_to_num((self.edge_accum / residual_denom).squeeze(1))
        footprint_denom = self.footprint_denom.clamp_min(1.0)
        footprint_residual = torch.nan_to_num(
            (self.footprint_residual_accum / footprint_denom).squeeze(1))
        footprint_edge = torch.nan_to_num((self.footprint_edge_accum / footprint_denom).squeeze(1))
        pixel_coverage = torch.nan_to_num((self.projected_area_accum / residual_denom).squeeze(1))
        valid = (self.denom.squeeze(1) > 0) & torch.isfinite(gradient)
        unique_views = bitset_popcount(
            self.view_direction_support, int(opt.densification_unique_view_bins))
        self.unique_view_support[:, 0] = unique_views
        sky_support = torch.nan_to_num((self.sky_support / residual_denom).squeeze(1)).clamp(0.0, 1.0)
        low_parallax_support = torch.nan_to_num(
            (self.low_parallax_support / residual_denom).squeeze(1)).clamp(0.0, 1.0)
        depth_support = torch.nan_to_num(
            (self.depth_consistent_support / residual_denom).squeeze(1)).clamp(0.0, 1.0)
        persistent_mode = opt.densification_method == "persistent_multiview_hybrid"
        persistent_scores = None
        if persistent_mode:
            persistent_scores = compute_persistent_scores(
                original_grad=gradient, abs_grad=absolute_gradient, grad_sq=(self.xyz_gradient_sq_accum / denom).squeeze(1),
                residual=residual, edge=edge, unique_views=unique_views, depth_support=depth_support,
                sky_support=sky_support, low_parallax_support=low_parallax_support,
                valid_mask=valid, cfg=opt, footprint_residual=footprint_residual,
                footprint_edge=footprint_edge, pixel_coverage=pixel_coverage)
            score = persistent_scores.total
            if opt.densification_selection_mode == "percentile":
                window_hit = percentile_mask(score, valid, opt.densification_percentile)
                threshold = (float(torch.quantile(score[valid], opt.densification_percentile).item())
                             if valid.any() else float("inf"))
            else:
                threshold = float(opt.densification_score_threshold)
                window_hit = valid & (score >= threshold)
            window_hit &= self.denom.squeeze(1) >= int(opt.densification_min_visibility_count)
            window_hit &= self.gaussian_age.squeeze(1) >= int(opt.min_gaussian_age)
            if opt.densification_unique_view_support_enabled:
                window_hit &= unique_views >= int(opt.densification_min_unique_view_bins)
            if opt.densification_require_depth_consistency:
                window_hit &= depth_support >= float(opt.densification_min_depth_support)
            update_persistent_window(
                self.persistent_score_ema, self.persistent_hit_ema, self.window_hit_count,
                self.recent_window_mask, window_hit, score,
                float(opt.densification_persistence_decay), int(opt.densification_recent_window_count))
            self.gradient_burstiness[:, 0] = persistent_scores.burstiness
            self.persistent_edge_ema.mul_(float(opt.densification_persistence_decay)).add_(
                edge[:, None], alpha=1.0 - float(opt.densification_persistence_decay))
            eligible = (
                (recent_hit_count(self.recent_window_mask, int(opt.densification_recent_window_count))
                 >= int(opt.densification_min_recent_hits))
                & (self.window_hit_count.squeeze(1) >= int(opt.densification_min_persistent_windows))
                & (self.persistent_score_ema.squeeze(1) >= float(opt.densification_persistent_threshold))
            )
        else:
            score = self.compute_densification_score(
                gradient, absolute_gradient,
                residual if opt.densification_residual_aware else None,
                edge if opt.densification_edge_aware else None,
                rms_gradient, opt, valid)
            if opt.densification_selection_mode == "percentile":
                eligible = percentile_mask(score, valid, opt.densification_percentile)
                threshold = float(torch.quantile(score[valid], opt.densification_percentile).item()) if valid.any() else float("inf")
            else:
                if opt.densification_method in ("absolute_gradient", "absolute_gradient_approx"):
                    threshold = float(opt.densification_abs_grad_threshold)
                elif opt.densification_method in ("hybrid", "residual"):
                    threshold = float(opt.densification_score_threshold)
                else:
                    threshold = float(max_grad)
                eligible = valid & (score >= threshold)
            eligible &= self.visibility_count.squeeze(1) >= int(opt.densification_min_visibility_count)
            eligible &= self.gaussian_age.squeeze(1) >= int(opt.min_gaussian_age)

        if not persistent_mode and opt.densification_selection_mode != "percentile":
            pass
        elif not persistent_mode:
            # Percentile path receives the common eligibility constraints here.
            eligible &= self.visibility_count.squeeze(1) >= int(opt.densification_min_visibility_count)
            eligible &= self.gaussian_age.squeeze(1) >= int(opt.min_gaussian_age)
        budget = min(max(0, int(opt.max_gaussians) - self.get_xyz.shape[0]),
                     max(0, int(opt.max_new_gaussians_per_step)))
        self._last_spatial_selection = None
        if getattr(opt, "spatial_budget_enabled", False) and self._last_spatial_context is not None and budget > 0:
            context = self._last_spatial_context
            if context["centers"].shape[0] == eligible.shape[0]:
                spatial = spatially_balanced_topk(
                    eligible & context["visible"], score, context["centers"], context["tile_energy"],
                    context["image_width"], context["image_height"], budget,
                    int(opt.spatial_tile_size), float(opt.spatial_tile_budget_gamma),
                    int(opt.spatial_tile_min_new_gaussians), int(opt.spatial_tile_max_new_gaussians))
                selected = spatial.selected
                remaining = max(0, budget - int(selected.sum().item()))
                if remaining:
                    selected |= self._limit_mask(eligible & ~selected, score, remaining)
                eligible = selected
                self._last_spatial_selection = spatial
        if persistent_mode:
            self.last_densified_iteration[eligible, 0] = int(iteration)
        metric_gradient = gradient
        small = self.get_scaling.max(dim=1).values <= self.percent_dense * extent
        clone_mask, split_mask = eligible & small, eligible & ~small
        clone_mask = self._limit_mask(clone_mask, score, budget)
        budget -= int(clone_mask.sum().item())
        # A split temporarily appends two children before pruning its parent.
        # Reserve both slots so MAX_GAUSSIANS is respected even mid-operation.
        split_mask = self._limit_mask(split_mask, score, budget // 2)
        self._density_control_metrics = {}
        self._structure_split_metrics = {}
        cloned = self.densify_and_clone(
            gradient.unsqueeze(1), max_grad, extent, clone_mask, options=opt)
        if cloned:
            gradient = torch.cat((gradient, torch.zeros(cloned, device=gradient.device)))
            split_mask = torch.cat((split_mask, torch.zeros(cloned, device=split_mask.device, dtype=torch.bool)))
        split = self.densify_and_split(
            gradient.unsqueeze(1), max_grad, extent, 2, split_mask, options=opt)
        if getattr(opt, "densification_opacity_pruning_enabled", True):
            prune_mask = conservative_opacity_prune_mask(
                self.get_opacity, self.visibility_count, self.gaussian_age,
                min_opacity, int(getattr(opt, "min_gaussian_age", 200)))
        else:
            # Opacity resets and opacity-conserving splits temporarily produce
            # many low-alpha children. V5 delegates their removal to scheduled
            # importance pruning, which has visibility and support evidence.
            prune_mask = torch.zeros(
                self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        thin_protected = self.thin_structure_mask(opt, iteration)
        prune_mask &= ~thin_protected
        if max_screen_size:
            if getattr(opt, "densification_screen_size_pruning_enabled", True):
                prune_mask |= (self.max_radii2D > max_screen_size)
            prune_mask |= (self.get_scaling.max(dim=1).values > 0.1 * extent)
        pruned = int(prune_mask.sum().item())
        if pruned and pruned < self.get_xyz.shape[0]:
            self.prune_points(prune_mask)
        self.tmp_radii = None
        self._last_densify_counts = {"cloned": cloned, "split": split, "pruned": pruned}
        signal_counts = {}
        if persistent_scores is not None:
            signal_masks = {
                "abs_grad": percentile_mask(persistent_scores.absolute_gradient, valid, opt.densification_percentile)
                            & (persistent_scores.absolute_gradient > 0),
                "residual": percentile_mask(persistent_scores.residual, valid, opt.densification_percentile)
                            & (persistent_scores.residual > 0),
                "edge": percentile_mask(persistent_scores.edge, valid, opt.densification_percentile)
                            & (persistent_scores.edge > 0),
                "multiview": percentile_mask(persistent_scores.multiview, valid, opt.densification_percentile)
                            & (persistent_scores.multiview > 0),
            }
            signal_counts = {f"selected_by_{name}": int(mask.sum().item())
                             for name, mask in signal_masks.items()}
            signal_counts["signal_intersection"] = int(torch.stack(list(signal_masks.values())).all(dim=0).sum().item())
        self._last_densification_metrics = {
            "original_grad_mean": float(metric_gradient[valid].mean().item()) if valid.any() else 0.0,
            "abs_grad_mean": float(absolute_gradient[valid].mean().item()) if valid.any() else 0.0,
            "residual_corrected_mean": float(residual[valid].mean().item()) if valid.any() else 0.0,
            "hybrid_score_mean": float(score[valid].mean().item()) if valid.any() else 0.0,
            "selected_count": int(eligible.sum().item()), "threshold": threshold,
            "max_count_reached": int(self.get_xyz.shape[0] >= int(opt.max_gaussians)),
            "unique_view_support_mean": float(unique_views.float().mean().item()) if unique_views.numel() else 0.0,
            "burstiness_mean": (float(persistent_scores.burstiness[valid].mean().item())
                                if persistent_scores is not None and valid.any() else 0.0),
            "persistent_hit_mean": (float(self.persistent_hit_ema.mean().item()) if persistent_mode else 0.0),
            "selected_in_sky": int((eligible & (sky_support > 0.5)).sum().item()),
            "selected_in_low_parallax": int((eligible & (low_parallax_support > 0.5)).sum().item()),
            "thin_protected": int(thin_protected.sum().item()),
            "thin_protection_rate": float(thin_protected.float().mean().item()),
            "footprint_residual_mean": (float(footprint_residual[valid].mean().item())
                                        if valid.any() else 0.0),
            "footprint_edge_mean": (float(footprint_edge[valid].mean().item())
                                    if valid.any() else 0.0),
            "footprint_sampled_fraction": float((self.footprint_denom.squeeze(1) > 0).float().mean().item()),
            "spatial_selected_count": (int(self._last_spatial_selection.selected.sum().item())
                                       if self._last_spatial_selection is not None else 0),
            "spatial_budget_total": (int(self._last_spatial_selection.tile_budget.sum().item())
                                     if self._last_spatial_selection is not None else 0),
            "spatial_top_decile_budget_fraction": (
                self._last_spatial_selection.top_decile_budget_fraction
                if self._last_spatial_selection is not None else 0.0),
            **self._density_control_metrics,
            **self._structure_split_metrics,
            **signal_counts,
        }
        self._reset_interval_statistics()

    def _reset_interval_statistics(self):
        for name in ("xyz_gradient_accum", "xyz_gradient_abs_accum", "xyz_gradient_sq_accum",
                     "denom", "residual_accum", "residual_denom", "edge_accum",
                     "depth_consistent_support", "sky_support", "low_parallax_support",
                     "view_direction_support", "footprint_residual_accum",
                     "footprint_edge_accum", "footprint_denom"):
            value = getattr(self, name)
            if value.numel():
                value.zero_()
        if self.max_radii2D.numel():
            self.max_radii2D.zero_()

    def reset_image_space_statistics(self):
        """Reset resolution-dependent accumulators after a camera-scale change."""
        if self.densification_stats_enabled:
            self._reset_interval_statistics()

    def prune_low_importance(self, opt, iteration=0):
        if self.get_xyz.shape[0] <= 1:
            return 0
        visibility = self.visibility_count.float().clamp_min(1.0)
        importance = (self.importance_accum / visibility) * torch.log1p(self.visibility_count.float()) * (self.projected_area_accum / visibility)
        self.last_mean_importance = importance.mean().item()
        low = importance.squeeze(1) < float(opt.importance_pruning_threshold)
        weak = (self.get_opacity.squeeze(1) < float(opt.importance_pruning_min_opacity)) | (self.visibility_count.squeeze(1) < int(opt.importance_pruning_min_visibility_count))
        mask = low & weak & (self.gaussian_age.squeeze(1) > int(opt.min_gaussian_age))
        mask &= ~self.thin_structure_mask(opt, iteration)
        if mask.any() and (~mask).any():
            count = int(mask.sum().item())
            self.prune_points(mask)
            self._last_densify_counts["pruned"] += count
            return count
        return 0
