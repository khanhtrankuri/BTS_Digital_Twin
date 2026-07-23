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

from argparse import ArgumentParser, Namespace
import sys
import os


def load_bts_geogs_config(path):
    """Map a readable BTS-GeoGS YAML preset onto the existing argparse API."""
    from utils.config_utils import load_yaml_with_base
    config = load_yaml_with_base(path)
    mapping = {
        ("MODEL", "GEOMETRY_AWARE", "ENABLED"): "geometry_aware",
        ("MODEL", "GEOMETRY_AWARE", "DEPTH_LOSS_ENABLED"): "depth_loss_enabled",
        ("MODEL", "GEOMETRY_AWARE", "DEPTH_LOSS_WEIGHT"): "depth_loss_weight",
        ("MODEL", "GEOMETRY_AWARE", "DEPTH_CONFIDENCE_WEIGHTED"): "depth_confidence_weighted",
        ("MODEL", "GEOMETRY_AWARE", "DEPTH_MIN_CONFIDENCE"): "depth_min_confidence",
        ("MODEL", "GEOMETRY_AWARE", "NORMAL_LOSS_ENABLED"): "normal_loss_enabled",
        ("MODEL", "GEOMETRY_AWARE", "NORMAL_LOSS_WEIGHT"): "normal_loss_weight",
        ("MODEL", "GEOMETRY_AWARE", "NORMAL_USE_ABS_COSINE"): "normal_use_abs_cosine",
        ("MODEL", "GEOMETRY_AWARE", "EDGE_LOSS_ENABLED"): "edge_loss_enabled",
        ("MODEL", "GEOMETRY_AWARE", "EDGE_LOSS_WEIGHT"): "edge_loss_weight",
        ("MODEL", "GEOMETRY_AWARE", "EDGE_WEIGHT_GAMMA"): "edge_weight_gamma",
        ("MODEL", "GEOMETRY_AWARE", "SCALE_REG_ENABLED"): "scale_reg_enabled",
        ("MODEL", "GEOMETRY_AWARE", "SCALE_REG_WEIGHT"): "scale_reg_weight",
        ("MODEL", "GEOMETRY_AWARE", "MAX_GAUSSIAN_SCALE"): "max_gaussian_scale",
        ("MODEL", "GEOMETRY_AWARE", "MAX_ANISOTROPY_RATIO"): "max_anisotropy_ratio",
        ("DENSIFICATION", "EDGE_AWARE"): "densification_edge_aware",
        ("DENSIFICATION", "RESIDUAL_AWARE"): "densification_residual_aware",
        ("DENSIFICATION", "GRADIENT_WEIGHT"): "densification_gradient_weight",
        ("DENSIFICATION", "RESIDUAL_WEIGHT"): "densification_residual_weight",
        ("DENSIFICATION", "EDGE_WEIGHT"): "densification_edge_weight",
        ("DENSIFICATION", "METHOD"): "densification_method",
        ("DENSIFICATION", "ORIGINAL_GRAD_WEIGHT"): "densification_original_grad_weight",
        ("DENSIFICATION", "ABS_GRAD_WEIGHT"): "densification_abs_grad_weight",
        ("DENSIFICATION", "ABS_GRAD_THRESHOLD"): "densification_abs_grad_threshold",
        ("DENSIFICATION", "RESIDUAL_TYPE"): "densification_residual_type",
        ("DENSIFICATION", "SELECTION_MODE"): "densification_selection_mode",
        ("DENSIFICATION", "PERCENTILE"): "densification_percentile",
        ("DENSIFICATION", "SCORE_THRESHOLD"): "densification_score_threshold",
        ("DENSIFICATION", "MIN_VISIBILITY_COUNT"): "densification_min_visibility_count",
        ("DENSIFICATION", "MAX_NEW_GAUSSIANS_PER_STEP"): "max_new_gaussians_per_step",
        ("DENSIFICATION", "FROM_ITER"): "densify_from_iter",
        ("DENSIFICATION", "UNTIL_ITER"): "densify_until_iter",
        ("DENSIFICATION", "INTERVAL"): "densification_interval",
        ("DENSIFICATION", "MAX_GAUSSIANS"): "max_gaussians",
        ("DENSIFICATION", "MIN_GAUSSIAN_AGE"): "min_gaussian_age",
        ("DENSIFICATION", "WINDOW_SIZE"): "densification_window_size",
        ("DENSIFICATION", "PERSISTENCE_DECAY"): "densification_persistence_decay",
        ("DENSIFICATION", "MIN_PERSISTENT_WINDOWS"): "densification_min_persistent_windows",
        ("DENSIFICATION", "PERSISTENT_THRESHOLD"): "densification_persistent_threshold",
        ("DENSIFICATION", "RECENT_WINDOW_COUNT"): "densification_recent_window_count",
        ("DENSIFICATION", "MIN_RECENT_HITS"): "densification_min_recent_hits",
        ("DENSIFICATION", "UNIQUE_VIEW_SUPPORT_ENABLED"): "densification_unique_view_support_enabled",
        ("DENSIFICATION", "UNIQUE_VIEW_BINS"): "densification_unique_view_bins",
        ("DENSIFICATION", "MIN_UNIQUE_VIEW_BINS"): "densification_min_unique_view_bins",
        ("DENSIFICATION", "MULTIVIEW_WEIGHT"): "densification_multiview_weight",
        ("DENSIFICATION", "DEPTH_SUPPORT_WEIGHT"): "densification_depth_support_weight",
        ("DENSIFICATION", "REQUIRE_DEPTH_CONSISTENCY"): "densification_require_depth_consistency",
        ("DENSIFICATION", "MIN_DEPTH_SUPPORT"): "densification_min_depth_support",
        ("DENSIFICATION", "BURST_SUPPRESSION_ENABLED"): "densification_burst_suppression_enabled",
        ("DENSIFICATION", "BURST_PENALTY_WEIGHT"): "densification_burst_penalty_weight",
        ("DENSIFICATION", "SKY_SCORE_MULTIPLIER"): "densification_sky_score_multiplier",
        ("DENSIFICATION", "LOW_PARALLAX_SCORE_MULTIPLIER"): "densification_low_parallax_score_multiplier",
        ("DENSIFICATION", "OPACITY_PRUNING_ENABLED"): "densification_opacity_pruning_enabled",
        ("DENSIFICATION", "SCREEN_SIZE_PRUNING_ENABLED"): "densification_screen_size_pruning_enabled",
        ("DENSIFICATION", "OPACITY_CONSERVING_CLONE"): "opacity_conserving_clone",
        ("DENSIFICATION", "OPACITY_CONSERVING_SPLIT"): "opacity_conserving_split",
        ("DENSIFICATION", "STRUCTURE_ALIGNED_SPLIT"): "structure_aligned_split_enabled",
        ("DENSIFICATION", "WIRE_RATIO_THRESHOLD"): "wire_ratio_threshold",
        ("DENSIFICATION", "SURFACE_RATIO_THRESHOLD"): "surface_ratio_threshold",
        ("DENSIFICATION", "WIRE_SPLIT_OFFSET"): "wire_split_offset",
        ("DENSIFICATION", "WIRE_MAJOR_SCALE_FACTOR"): "wire_major_scale_factor",
        ("DENSIFICATION", "WIRE_MINOR_SCALE_FACTOR"): "wire_minor_scale_factor",
        ("DENSIFICATION", "SURFACE_SPLIT_OFFSET"): "surface_split_offset",
        ("DENSIFICATION", "BLOB_RANDOM_SPLIT"): "blob_random_split",
        ("DENSIFICATION", "FOOTPRINT_SAMPLING_ENABLED"): "footprint_sampling_enabled",
        ("DENSIFICATION", "FOOTPRINT_PATTERN"): "footprint_pattern",
        ("DENSIFICATION", "FOOTPRINT_MEAN_WEIGHT"): "footprint_mean_weight",
        ("DENSIFICATION", "FOOTPRINT_P90_WEIGHT"): "footprint_p90_weight",
        ("DENSIFICATION", "FOOTPRINT_MAX_WEIGHT"): "footprint_max_weight",
        ("DENSIFICATION", "FOOTPRINT_MIN_SCREEN_RADIUS"): "footprint_min_screen_radius",
        ("DENSIFICATION", "CENTER_RESIDUAL_WEIGHT"): "center_residual_weight",
        ("DENSIFICATION", "FOOTPRINT_RESIDUAL_WEIGHT"): "footprint_residual_weight",
        ("DENSIFICATION", "FOOTPRINT_EDGE_WEIGHT"): "footprint_edge_weight",
        ("DENSIFICATION", "PIXEL_COVERAGE_WEIGHT"): "pixel_coverage_weight",
        ("DENSIFICATION", "SPATIAL_BUDGET_ENABLED"): "spatial_budget_enabled",
        ("DENSIFICATION", "TILE_SIZE"): "spatial_tile_size",
        ("DENSIFICATION", "TILE_BUDGET_GAMMA"): "spatial_tile_budget_gamma",
        ("DENSIFICATION", "TILE_MIN_NEW_GAUSSIANS"): "spatial_tile_min_new_gaussians",
        ("DENSIFICATION", "TILE_MAX_NEW_GAUSSIANS"): "spatial_tile_max_new_gaussians",
        ("DENSIFICATION", "TILE_EDGE_WEIGHT"): "spatial_tile_edge_weight",
        ("PRUNING", "IMPORTANCE_ENABLED"): "importance_pruning_enabled",
        ("PRUNING", "START_ITER"): "importance_pruning_start_iter",
        ("PRUNING", "INTERVAL"): "importance_pruning_interval",
        ("PRUNING", "MIN_OPACITY"): "importance_pruning_min_opacity",
        ("PRUNING", "MIN_VISIBILITY_COUNT"): "importance_pruning_min_visibility_count",
        ("PRUNING", "IMPORTANCE_THRESHOLD"): "importance_pruning_threshold",
        ("PRUNING", "THIN_STRUCTURE_PROTECTION"): "thin_structure_protection",
        ("PRUNING", "THIN_MIN_ANISOTROPY"): "thin_min_anisotropy",
        ("PRUNING", "THIN_MIN_EDGE_SUPPORT"): "thin_min_edge_support",
        ("PRUNING", "THIN_MIN_VIEW_BINS"): "thin_min_view_bins",
        ("PRUNING", "THIN_MAX_PROJECTED_AREA"): "thin_max_projected_area",
        ("PRUNING", "THIN_PROTECTION_DECAY_AFTER_ITER"): "thin_protection_decay_after_iter",
        ("INITIALIZATION", "MODE"): "initialization_mode",
        ("INITIALIZATION", "CAP_TO_MAX_GAUSSIANS"): "initialization_cap_to_max_gaussians",
        ("INITIALIZATION", "DENSE_PRIOR_PATH"): "dense_prior_path",
        ("INITIALIZATION", "CONFIDENCE_THRESHOLD"): "dense_prior_confidence_threshold",
        ("INITIALIZATION", "VOXEL_SIZE"): "dense_prior_voxel_size",
        ("INITIALIZATION", "KNN_K"): "dense_prior_knn_k",
        ("INITIALIZATION", "INITIALIZE_ROTATION_FROM_NORMAL"): "initialize_rotation_from_normal",
        ("INITIALIZATION", "INITIALIZE_OPACITY_FROM_CONFIDENCE"): "initialize_opacity_from_confidence",
        ("EXPOSURE", "ENABLED"): "exposure_compensation",
        ("EXPOSURE", "MODE"): "exposure_mode",
        ("EXPOSURE", "START_ITER"): "exposure_start_iter",
        ("EXPOSURE", "END_ITER"): "exposure_end_iter",
        ("EXPOSURE", "FREEZE_ITER"): "exposure_freeze_iter",
        ("EXPOSURE", "LR_INIT"): "exposure_lr_init",
        ("EXPOSURE", "LR_FINAL"): "exposure_lr_final",
        ("EXPOSURE", "MIN_GAIN"): "exposure_min_gain",
        ("EXPOSURE", "MAX_GAIN"): "exposure_max_gain",
        ("EXPOSURE", "MAX_BIAS"): "exposure_max_bias",
        ("EXPOSURE", "GAIN_REG_WEIGHT"): "exposure_gain_reg_weight",
        ("EXPOSURE", "BIAS_REG_WEIGHT"): "exposure_bias_reg_weight",
        ("EXPOSURE", "ZERO_MEAN_REG_WEIGHT"): "exposure_zero_mean_reg_weight",
        ("EXPOSURE", "TEST_MODE"): "test_exposure_mode",
        ("EXPOSURE", "TEST_K"): "exposure_test_k",
        ("EXPOSURE", "POSITION_WEIGHT"): "exposure_position_weight",
        ("EXPOSURE", "ANGLE_WEIGHT"): "exposure_angle_weight",
        ("EXPOSURE", "TEMPORAL_WEIGHT"): "exposure_temporal_weight",
        ("EXPOSURE", "FOCAL_WEIGHT"): "exposure_focal_weight",
        ("EXPOSURE", "DISTANCE_TEMPERATURE"): "exposure_distance_temperature",
        ("EXPOSURE", "CONFIDENCE_TEMPERATURE"): "exposure_confidence_temperature",
        ("EXPOSURE", "MIN_CONFIDENCE"): "exposure_min_confidence",
        ("EXPOSURE", "MAX_GAIN_DELTA_AT_TEST"): "exposure_max_gain_delta_at_test",
        ("EXPOSURE", "MAX_BIAS_AT_TEST"): "exposure_max_bias_at_test",
        ("EXPOSURE", "FIELD_MODE"): "exposure_field_mode",
        ("EXPOSURE", "SPLINE_ENABLED"): "exposure_spline_enabled",
        ("EXPOSURE", "SPLINE_NUM_KNOTS"): "exposure_spline_num_knots",
        ("EXPOSURE", "SPLINE_DEGREE"): "exposure_spline_degree",
        ("EXPOSURE", "SPLINE_SMOOTHNESS_WEIGHT"): "exposure_spline_smoothness_weight",
        ("EXPOSURE", "PER_VIEW_RESIDUAL_ENABLED"): "exposure_per_view_residual_enabled",
        ("EXPOSURE", "PER_VIEW_RESIDUAL_WEIGHT"): "exposure_per_view_residual_weight",
        ("CAMERA", "USE_UNDISTORTED_DATA"): "camera_use_undistorted_data",
        ("CAMERA", "STRICT_MODEL_VALIDATION"): "camera_strict_model_validation",
        ("CAMERA", "WARN_IF_DISTORTION_DROPPED"): "camera_warn_if_distortion_dropped",
        ("CAMERA", "REQUIRE_UNDISTORTION_METADATA"): "camera_require_undistortion_metadata",
        ("SH_SCHEDULE", "ENABLED"): "sh_schedule_enabled",
        ("SH_SCHEDULE", "MILESTONES"): "sh_schedule_milestones",
        ("RESOLUTION_SCHEDULE", "ENABLED"): "resolution_schedule_enabled",
        ("RESOLUTION_SCHEDULE", "STAGES"): "resolution_schedule_stages",
        ("RESOLUTION_SCHEDULE", "CACHE_ON_CPU"): "resolution_cache_on_cpu",
        ("DATA", "RESOLUTION"): "resolution",
        ("SKY", "ENABLED"): "sky_enabled",
        ("SKY", "MASK_DIR"): "sky_mask_dir",
        ("SKY", "PHOTOMETRIC_WEIGHT"): "sky_photometric_weight",
        ("SKY", "DEPTH_WEIGHT"): "sky_depth_weight",
        ("SKY", "NORMAL_WEIGHT"): "sky_normal_weight",
        ("SKY", "EDGE_WEIGHT"): "sky_edge_weight",
        ("SKY", "DENSIFICATION_WEIGHT"): "densification_sky_score_multiplier",
        ("SKY", "BACKGROUND_MODE"): "sky_background_mode",
        ("SKY", "BACKGROUND_DEGREE"): "sky_background_degree",
        ("SKY", "BACKGROUND_LR"): "sky_background_lr",
        ("LOW_PARALLAX", "ENABLED"): "low_parallax_enabled",
        ("LOW_PARALLAX", "MASK_DIR"): "low_parallax_mask_dir",
        ("LOW_PARALLAX", "GEOMETRY_WEIGHT"): "low_parallax_geometry_weight",
        ("LOW_PARALLAX", "DENSIFICATION_WEIGHT"): "densification_low_parallax_score_multiplier",
        ("IMAGE_QUALITY", "SHARPNESS_AWARE_SAMPLING"): "sharpness_aware_sampling",
        ("IMAGE_QUALITY", "MIN_SAMPLE_WEIGHT"): "sharpness_min_sample_weight",
        ("IMAGE_QUALITY", "MAX_SAMPLE_WEIGHT"): "sharpness_max_sample_weight",
        ("IMAGE_QUALITY", "BLUR_EDGE_WEIGHT_MIN"): "blur_edge_weight_min",
        ("IMAGE_QUALITY", "USE_CHARBONNIER"): "use_charbonnier",
        ("IMAGE_QUALITY", "CHARBONNIER_EPS"): "charbonnier_eps",
        ("PERCEPTUAL", "ENABLED"): "perceptual_loss_enabled",
        ("PERCEPTUAL", "START_ITER"): "perceptual_loss_start_iter",
        ("PERCEPTUAL", "END_ITER"): "perceptual_loss_end_iter",
        ("PERCEPTUAL", "INTERVAL"): "perceptual_loss_interval",
        ("PERCEPTUAL", "WEIGHT"): "perceptual_loss_weight",
        ("PERCEPTUAL", "MAX_SIZE"): "perceptual_loss_max_size",
        ("MULTIVIEW_DEPTH", "ENABLED"): "multiview_depth_enabled",
        ("MULTIVIEW_DEPTH", "WEIGHT"): "multiview_depth_weight",
        ("MULTIVIEW_DEPTH", "INTERVAL"): "multiview_depth_interval",
        ("MULTIVIEW_DEPTH", "SIGMA_Z"): "multiview_depth_sigma",
        ("MULTIVIEW_DEPTH", "RELATIVE_THRESHOLD"): "multiview_depth_relative_threshold",
        ("MULTIVIEW_RGB", "ENABLED"): "multiview_rgb_enabled",
        ("MULTIVIEW_RGB", "START_ITER"): "multiview_start_iter",
        ("MULTIVIEW_RGB", "PEAK_ITER"): "multiview_peak_iter",
        ("MULTIVIEW_RGB", "END_ITER"): "multiview_end_iter",
        ("MULTIVIEW_RGB", "INTERVAL"): "multiview_interval",
        ("MULTIVIEW_RGB", "WEIGHT_INITIAL"): "multiview_weight_initial",
        ("MULTIVIEW_RGB", "WEIGHT_PEAK"): "multiview_weight_peak",
        ("MULTIVIEW_RGB", "WEIGHT_FINAL"): "multiview_weight_final",
        ("MULTIVIEW_RGB", "LOSS_MODE"): "multiview_loss_mode",
        ("MULTIVIEW_RGB", "ZNCC_WEIGHT"): "multiview_zncc_weight",
        ("MULTIVIEW_RGB", "GRADIENT_WEIGHT"): "multiview_gradient_weight",
        ("MULTIVIEW_RGB", "CHARBONNIER_WEIGHT"): "multiview_charbonnier_weight",
        ("MULTIVIEW_RGB", "PATCH_SIZE"): "multiview_patch_size",
        ("MULTIVIEW_RGB", "SIGMA_Z"): "multiview_sigma_z",
        ("MULTIVIEW_RGB", "RELATIVE_DEPTH_THRESHOLD"): "multiview_relative_depth_threshold",
        ("MULTIVIEW_RGB", "MIN_VALID_PIXELS"): "multiview_min_valid_pixels",
        ("MULTIVIEW_RGB", "MIN_ALPHA"): "multiview_min_alpha",
        ("MULTIVIEW_RGB", "DETACH_SOURCE_DEPTH"): "multiview_detach_source_depth",
        ("MULTIVIEW_RGB", "NUM_SOURCE_VIEWS"): "multiview_num_source_views",
        ("MULTIVIEW_RGB", "POSITION_WEIGHT"): "multiview_position_weight",
        ("MULTIVIEW_RGB", "ANGLE_WEIGHT"): "multiview_angle_weight",
        ("MULTIVIEW_RGB", "OVERLAP_WEIGHT"): "multiview_overlap_weight",
        ("MULTIVIEW_RGB", "SHARPNESS_WEIGHT"): "multiview_sharpness_weight",
        ("MULTIVIEW_RGB", "TEMPORAL_WEIGHT"): "multiview_temporal_weight",
        ("MULTIVIEW_RGB", "EXPOSURE_MISMATCH_WEIGHT"): "multiview_exposure_mismatch_weight",
        ("MULTIVIEW_RGB", "MIN_OVERLAP"): "multiview_min_overlap",
        ("MULTIVIEW_RGB", "TEMPORAL_BRACKETING"): "multiview_temporal_bracketing",
        ("MULTIVIEW_RGB", "BLUR_SOURCE_THRESHOLD"): "multiview_blur_source_threshold",
        ("MULTIVIEW_RGB", "SHARP_EQUIVALENCE_MARGIN"): "multiview_sharp_equivalence_margin",
        ("RAY_DEPTH_VARIANCE", "ENABLED"): "ray_depth_variance_enabled",
        ("RAY_DEPTH_VARIANCE", "START_ITER"): "ray_depth_variance_start_iter",
        ("RAY_DEPTH_VARIANCE", "PEAK_ITER"): "ray_depth_variance_peak_iter",
        ("RAY_DEPTH_VARIANCE", "END_ITER"): "ray_depth_variance_end_iter",
        ("RAY_DEPTH_VARIANCE", "WEIGHT_INITIAL"): "ray_depth_variance_weight_initial",
        ("RAY_DEPTH_VARIANCE", "WEIGHT_PEAK"): "ray_depth_variance_weight_peak",
        ("RAY_DEPTH_VARIANCE", "WEIGHT_FINAL"): "ray_depth_variance_weight_final",
        ("RAY_DEPTH_VARIANCE", "MIN_ALPHA"): "ray_depth_variance_min_alpha",
        ("RAY_DEPTH_VARIANCE", "MAX_UNCERTAINTY"): "ray_depth_variance_max_uncertainty",
        ("RAY_DEPTH_VARIANCE", "EDGE_WEIGHT_MULTIPLIER"): "ray_depth_variance_edge_weight_multiplier",
        ("RAY_DEPTH_VARIANCE", "THIN_STRUCTURE_WEIGHT_MULTIPLIER"): "ray_depth_variance_thin_weight_multiplier",
        ("POSE_REFINEMENT", "ENABLED"): "pose_refinement_enabled",
        ("POSE_REFINEMENT", "START_ITER"): "pose_refinement_start_iter",
        ("POSE_REFINEMENT", "FREEZE_ITER"): "pose_refinement_freeze_iter",
        ("POSE_REFINEMENT", "ROTATION_LR"): "pose_rotation_lr",
        ("POSE_REFINEMENT", "TRANSLATION_LR"): "pose_translation_lr",
        ("POSE_REFINEMENT", "MAX_ROTATION_DEG"): "pose_max_rotation_deg",
        ("POSE_REFINEMENT", "MAX_TRANSLATION_RADIUS_RATIO"): "pose_max_translation_radius_ratio",
        ("POSE_REFINEMENT", "ROTATION_REG_WEIGHT"): "pose_rotation_reg_weight",
        ("POSE_REFINEMENT", "TRANSLATION_REG_WEIGHT"): "pose_translation_reg_weight",
        ("POSE_REFINEMENT", "TRAJECTORY_SMOOTHNESS_WEIGHT"): "pose_trajectory_smoothness_weight",
        ("BLUR_FORMATION", "ENABLED"): "blur_formation_enabled",
        ("BLUR_FORMATION", "NUM_SUBPOSES"): "blur_num_subposes",
        ("BLUR_FORMATION", "START_ITER"): "blur_start_iter",
        ("BLUR_FORMATION", "FREEZE_ITER"): "blur_freeze_iter",
        ("BLUR_FORMATION", "SHARPNESS_PERCENTILE_THRESHOLD"): "blur_sharpness_percentile_threshold",
        ("BLUR_FORMATION", "MAX_ROTATION_DEG"): "blur_max_rotation_deg",
        ("BLUR_FORMATION", "MAX_TRANSLATION_RADIUS_RATIO"): "blur_max_translation_radius_ratio",
        ("BLUR_FORMATION", "TRAJECTORY_REG_WEIGHT"): "blur_trajectory_reg_weight",
        ("BLUR_FORMATION", "DENSIFICATION_WEIGHT"): "blur_densification_weight",
        ("BLUR_FORMATION", "EDGE_WEIGHT"): "blur_formation_edge_weight",
        ("PATCH_REFINEMENT", "ENABLED"): "patch_refinement_enabled",
        ("PATCH_REFINEMENT", "START_ITER"): "patch_refinement_start_iter",
        ("PATCH_REFINEMENT", "PATCH_SIZE"): "patch_refinement_patch_size",
        ("PATCH_REFINEMENT", "RANDOM_RATIO"): "patch_refinement_random_ratio",
        ("PATCH_REFINEMENT", "RESIDUAL_RATIO"): "patch_refinement_residual_ratio",
        ("PATCH_REFINEMENT", "EDGE_RATIO"): "patch_refinement_edge_ratio",
        ("PATCH_REFINEMENT", "MIN_PATCH_DISTANCE"): "patch_refinement_min_patch_distance",
        ("VALIDATION", "SPLIT_FILE"): "validation_split_file",
        ("VALIDATION", "STRICT_SPARSE_PATH"): "strict_sparse_path",
    }
    defaults = dict(config.get("OPTIMIZATION", {}))
    defaults.update(config.get("PIPELINE", {}))
    for key_path, argument in mapping.items():
        node = config
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
        else:
            defaults[argument] = node
    return defaults

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.cache_images_on_cpu = False
        self.eval = False
        # Optional per-image geometry priors. Files are matched by image stem.
        self.depth_prior_dir = ""
        self.normal_prior_dir = ""
        self.confidence_prior_dir = ""
        self.camera_use_undistorted_data = False
        self.camera_strict_model_validation = True
        self.camera_warn_if_distortion_dropped = True
        self.camera_require_undistortion_metadata = False
        self.validation_split_file = ""
        self.strict_sparse_path = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        for name in ("validation_split_file", "strict_sparse_path"):
            value = getattr(g, name, "")
            if value and not os.path.isabs(value):
                setattr(g, name, os.path.abspath(os.path.join(g.source_path, value)))
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # BTS-GeoGS is deliberately opt-in. With geometry_aware=False the
        # original renderer, loss and densification paths are unchanged.
        self.geometry_aware = False
        self.depth_loss_enabled = False
        self.depth_loss_weight = 0.05
        self.depth_confidence_weighted = True
        self.depth_min_confidence = 0.3
        self.normal_loss_enabled = False
        self.normal_loss_weight = 0.02
        self.normal_use_abs_cosine = True
        self.edge_loss_enabled = False
        self.edge_loss_weight = 0.05
        self.edge_weight_gamma = 2.0
        self.scale_reg_enabled = False
        self.scale_reg_weight = 0.001
        self.max_gaussian_scale = 0.1
        self.max_anisotropy_ratio = 20.0
        self.geometry_warmup_until = 3000
        self.geometry_edge_warmup_factor = 0.0
        self.densification_edge_aware = False
        self.densification_residual_aware = False
        self.densification_gradient_weight = 1.0
        self.densification_residual_weight = 1.0
        self.densification_edge_weight = 1.0
        self.max_gaussians = 2_500_000
        self.min_gaussian_age = 300
        self.importance_pruning_enabled = False
        self.importance_pruning_start_iter = 8000
        self.importance_pruning_interval = 500
        self.importance_pruning_min_opacity = 0.005
        self.importance_pruning_min_visibility_count = 3
        self.importance_pruning_threshold = 0.0001
        self.initialization_mode = "colmap"
        self.initialization_cap_to_max_gaussians = False
        self.dense_prior_path = ""
        self.dense_prior_confidence_threshold = 0.3
        self.dense_prior_voxel_size = 0.01
        self.dense_prior_knn_k = 8
        self.initialize_rotation_from_normal = False
        self.initialize_opacity_from_confidence = False
        # BTS-GeoGS-v2 appearance and staged metric optimization (all opt-in).
        self.exposure_compensation = False
        self.exposure_mode = "diagonal_gain_bias"
        self.exposure_start_iter = 500
        self.exposure_end_iter = 30_000
        self.exposure_freeze_iter = 30_000
        self.exposure_matrix_reg_weight = 0.001
        self.exposure_gain_reg_weight = 0.001
        self.exposure_bias_reg_weight = 0.001
        self.exposure_zero_mean_reg_weight = 0.001
        self.exposure_min_gain = 0.75
        self.exposure_max_gain = 1.25
        self.exposure_max_bias = 0.10
        self.test_exposure_mode = "identity"
        self.exposure_test_k = 4
        self.exposure_position_weight = 1.0
        self.exposure_angle_weight = 0.5
        self.exposure_temporal_weight = 0.0
        self.exposure_focal_weight = 0.0
        self.exposure_distance_temperature = 0.10
        self.exposure_confidence_temperature = 0.08
        self.exposure_min_confidence = 0.0
        self.exposure_max_gain_delta_at_test = 0.20
        self.exposure_max_bias_at_test = 0.08
        self.exposure_field_mode = "per_view"
        self.exposure_spline_enabled = False
        self.exposure_spline_num_knots = 12
        self.exposure_spline_degree = 3
        self.exposure_spline_smoothness_weight = 0.01
        self.exposure_per_view_residual_enabled = True
        self.exposure_per_view_residual_weight = 0.01
        self.loss_schedule_enabled = False
        self.loss_stage_a_end = 12_000
        self.loss_stage_b_end = 30_000
        self.loss_stage_c_end = 45_000
        self.loss_stage_a_l1, self.loss_stage_a_mse, self.loss_stage_a_dssim = 0.70, 0.05, 0.20
        self.loss_stage_b_l1, self.loss_stage_b_mse, self.loss_stage_b_dssim = 0.40, 0.35, 0.25
        self.loss_stage_c_l1, self.loss_stage_c_mse, self.loss_stage_c_dssim = 0.15, 0.60, 0.25
        self.loss_stage_a_geometry, self.loss_stage_a_edge, self.loss_stage_a_exposure = 1.0, 0.5, 1.0
        self.loss_stage_b_geometry, self.loss_stage_b_edge, self.loss_stage_b_exposure = 0.2, 1.0, 1.0
        self.loss_stage_c_geometry, self.loss_stage_c_edge, self.loss_stage_c_exposure = 0.0, 0.2, 0.1
        # Optional late metric-aligned fine-tuning. Disabled by default so old
        # checkpoints and presets preserve their exact optimization path.
        self.perceptual_loss_enabled = False
        self.perceptual_loss_start_iter = 30_000
        self.perceptual_loss_end_iter = 32_000
        self.perceptual_loss_interval = 1
        self.perceptual_loss_weight = 0.05
        self.perceptual_loss_max_size = 256
        self.lr_stage_a_xyz, self.lr_stage_a_scaling, self.lr_stage_a_rotation = 1.0, 1.0, 1.0
        self.lr_stage_a_features, self.lr_stage_a_opacity, self.lr_stage_a_exposure = 1.0, 1.0, 1.0
        self.lr_stage_b_xyz, self.lr_stage_b_scaling, self.lr_stage_b_rotation = 0.5, 0.5, 0.5
        self.lr_stage_b_features, self.lr_stage_b_opacity, self.lr_stage_b_exposure = 1.0, 0.5, 0.5
        self.lr_stage_c_xyz, self.lr_stage_c_scaling, self.lr_stage_c_rotation = 0.1, 0.1, 0.1
        self.lr_stage_c_features, self.lr_stage_c_opacity, self.lr_stage_c_exposure = 0.5, 0.2, 0.1
        self.densification_method = "original"
        self.densification_abs_grad_weight = 1.0
        self.densification_original_grad_weight = 0.5
        self.densification_abs_grad_threshold = 0.0008
        self.densification_residual_type = "charbonnier"
        self.densification_selection_mode = "threshold"
        self.densification_percentile = 0.95
        self.densification_score_threshold = 1.0
        self.densification_min_visibility_count = 3
        self.max_new_gaussians_per_step = 100_000
        # BTS-GeoGS-v4 progressive schedules (disabled preserves v3 behavior).
        self.sh_schedule_enabled = False
        self.sh_schedule_milestones = [[0, 0], [3000, 1], [8000, 2], [18000, 3]]
        self.resolution_schedule_enabled = False
        self.resolution_schedule_stages = [
            {"START_ITER": 0, "SCALE": 0.50},
            {"START_ITER": 5000, "SCALE": 0.75},
            {"START_ITER": 15000, "SCALE": 1.00},
        ]
        self.resolution_cache_on_cpu = True
        self.densification_window_size = 100
        self.densification_persistence_decay = 0.80
        self.densification_min_persistent_windows = 3
        self.densification_persistent_threshold = 1.0
        self.densification_recent_window_count = 4
        self.densification_min_recent_hits = 3
        self.densification_unique_view_support_enabled = False
        self.densification_unique_view_bins = 12
        self.densification_min_unique_view_bins = 2
        self.densification_multiview_weight = 0.15
        self.densification_depth_support_weight = 0.10
        self.densification_require_depth_consistency = False
        self.densification_min_depth_support = 0.50
        self.densification_burst_suppression_enabled = False
        self.densification_burst_penalty_weight = 0.15
        self.densification_sky_score_multiplier = 0.05
        self.densification_low_parallax_score_multiplier = 0.20
        # Preserve the legacy behavior unless a newer config explicitly opts
        # out. Importance pruning remains a separate, multi-signal mechanism.
        self.densification_opacity_pruning_enabled = True
        self.densification_screen_size_pruning_enabled = True
        # BTS-GeoGS-v5 Phase A controls. They default off for old configs.
        self.opacity_conserving_clone = False
        self.opacity_conserving_split = False
        self.structure_aligned_split_enabled = False
        self.wire_ratio_threshold = 4.0
        self.surface_ratio_threshold = 4.0
        self.wire_split_offset = 0.35
        self.wire_major_scale_factor = 0.58
        self.wire_minor_scale_factor = 0.85
        self.surface_split_offset = 0.30
        self.blob_random_split = True
        self.footprint_sampling_enabled = False
        self.footprint_pattern = "9point"
        self.footprint_mean_weight = 0.50
        self.footprint_p90_weight = 0.40
        self.footprint_max_weight = 0.10
        self.footprint_min_screen_radius = 1.5
        self.center_residual_weight = 0.05
        self.footprint_residual_weight = 0.25
        self.footprint_edge_weight = 0.10
        self.pixel_coverage_weight = 0.10
        self.spatial_budget_enabled = False
        self.spatial_tile_size = 64
        self.spatial_tile_budget_gamma = 0.50
        self.spatial_tile_min_new_gaussians = 4
        self.spatial_tile_max_new_gaussians = 256
        self.spatial_tile_edge_weight = 0.25
        self.thin_structure_protection = False
        self.thin_min_anisotropy = 4.0
        self.thin_min_edge_support = 0.50
        self.thin_min_view_bins = 2
        self.thin_max_projected_area = 16.0
        self.thin_protection_decay_after_iter = 35_000
        self.sky_enabled = False
        self.sky_mask_dir = "masks/sky"
        self.sky_photometric_weight = 0.50
        self.sky_depth_weight = 0.0
        self.sky_normal_weight = 0.0
        self.sky_edge_weight = 0.05
        self.sky_background_mode = "constant"
        self.sky_background_degree = 2
        self.sky_background_lr = 0.001
        self.low_parallax_enabled = False
        self.low_parallax_mask_dir = "masks/low_parallax"
        self.low_parallax_geometry_weight = 0.20
        self.sharpness_aware_sampling = False
        self.sharpness_min_sample_weight = 0.50
        self.sharpness_max_sample_weight = 1.50
        self.blur_edge_weight_min = 0.20
        self.use_charbonnier = False
        self.charbonnier_eps = 0.001
        self.multiview_depth_enabled = False
        self.multiview_depth_weight = 0.01
        self.multiview_depth_interval = 10
        self.multiview_depth_sigma = 0.02
        self.multiview_depth_relative_threshold = 0.05
        self.multiview_rgb_enabled = False
        self.multiview_start_iter = 6000
        self.multiview_peak_iter = 18000
        self.multiview_end_iter = 35000
        self.multiview_interval = 5
        self.multiview_weight_initial = 0.0
        self.multiview_weight_peak = 0.05
        self.multiview_weight_final = 0.01
        self.multiview_loss_mode = "zncc_gradient"
        self.multiview_zncc_weight = 1.0
        self.multiview_gradient_weight = 0.20
        self.multiview_charbonnier_weight = 0.10
        self.multiview_patch_size = 7
        self.multiview_sigma_z = 0.02
        self.multiview_relative_depth_threshold = 0.05
        self.multiview_min_valid_pixels = 256
        self.multiview_min_alpha = 0.01
        self.multiview_detach_source_depth = True
        self.multiview_num_source_views = 2
        self.multiview_position_weight = 1.0
        self.multiview_angle_weight = 1.0
        self.multiview_overlap_weight = 1.0
        self.multiview_sharpness_weight = 0.5
        self.multiview_temporal_weight = 0.0
        self.multiview_exposure_mismatch_weight = 0.25
        self.multiview_min_overlap = 0.20
        self.multiview_temporal_bracketing = False
        self.multiview_blur_source_threshold = 0.35
        self.multiview_sharp_equivalence_margin = 0.10
        self.ray_depth_variance_enabled = False
        self.ray_depth_variance_start_iter = 4000
        self.ray_depth_variance_peak_iter = 16000
        self.ray_depth_variance_end_iter = 35000
        self.ray_depth_variance_weight_initial = 0.001
        self.ray_depth_variance_weight_peak = 0.010
        self.ray_depth_variance_weight_final = 0.002
        self.ray_depth_variance_min_alpha = 0.40
        self.ray_depth_variance_max_uncertainty = 0.60
        self.ray_depth_variance_edge_weight_multiplier = 0.50
        self.ray_depth_variance_thin_weight_multiplier = 0.30
        self.pose_refinement_enabled = False
        self.pose_refinement_start_iter = 2500
        self.pose_refinement_freeze_iter = 18000
        self.pose_rotation_lr = 0.000005
        self.pose_translation_lr = 0.000005
        self.pose_max_rotation_deg = 0.50
        self.pose_max_translation_radius_ratio = 0.005
        self.pose_rotation_reg_weight = 0.01
        self.pose_translation_reg_weight = 0.01
        self.pose_trajectory_smoothness_weight = 0.01
        self.blur_formation_enabled = False
        self.blur_num_subposes = 3
        self.blur_start_iter = 2000
        self.blur_freeze_iter = 18000
        self.blur_sharpness_percentile_threshold = 0.30
        self.blur_max_rotation_deg = 0.30
        self.blur_max_translation_radius_ratio = 0.003
        self.blur_trajectory_reg_weight = 0.01
        self.blur_densification_weight = 0.30
        self.blur_formation_edge_weight = 0.20
        self.patch_refinement_enabled = False
        self.patch_refinement_start_iter = 28000
        self.patch_refinement_patch_size = 512
        self.patch_refinement_random_ratio = 0.50
        self.patch_refinement_residual_ratio = 0.30
        self.patch_refinement_edge_ratio = 0.20
        self.patch_refinement_min_patch_distance = 128
        super().__init__(parser, "Optimization Parameters")


def load_model_config_defaults(model_path):
    """Return mapped YAML defaults saved with a trained model, if available."""

    config_path = os.path.join(str(model_path), "resolved_config.yaml")
    if not os.path.isfile(config_path):
        return None
    return load_bts_geogs_config(config_path)


def load_model_optimization_args(model_path):
    """Rebuild optimization-side models needed when rendering a v4/v5 PLY."""

    defaults = load_model_config_defaults(model_path)
    if defaults is None:
        return None
    parser = ArgumentParser(add_help=False)
    group = OptimizationParams(parser)
    parser.set_defaults(**defaults)
    return group.extract(parser.parse_args([]))

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
