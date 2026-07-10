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
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Using --config requires PyYAML. Install the environment from environment.yml.") from error
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
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
        ("DENSIFICATION", "FROM_ITER"): "densify_from_iter",
        ("DENSIFICATION", "UNTIL_ITER"): "densify_until_iter",
        ("DENSIFICATION", "INTERVAL"): "densification_interval",
        ("DENSIFICATION", "MAX_GAUSSIANS"): "max_gaussians",
        ("DENSIFICATION", "MIN_GAUSSIAN_AGE"): "min_gaussian_age",
        ("PRUNING", "IMPORTANCE_ENABLED"): "importance_pruning_enabled",
        ("PRUNING", "START_ITER"): "importance_pruning_start_iter",
        ("PRUNING", "INTERVAL"): "importance_pruning_interval",
        ("PRUNING", "MIN_OPACITY"): "importance_pruning_min_opacity",
        ("PRUNING", "MIN_VISIBILITY_COUNT"): "importance_pruning_min_visibility_count",
        ("PRUNING", "IMPORTANCE_THRESHOLD"): "importance_pruning_threshold",
        ("INITIALIZATION", "MODE"): "initialization_mode",
        ("INITIALIZATION", "DENSE_PRIOR_PATH"): "dense_prior_path",
        ("INITIALIZATION", "CONFIDENCE_THRESHOLD"): "dense_prior_confidence_threshold",
        ("INITIALIZATION", "VOXEL_SIZE"): "dense_prior_voxel_size",
        ("INITIALIZATION", "KNN_K"): "dense_prior_knn_k",
        ("INITIALIZATION", "INITIALIZE_ROTATION_FROM_NORMAL"): "initialize_rotation_from_normal",
        ("INITIALIZATION", "INITIALIZE_OPACITY_FROM_CONFIDENCE"): "initialize_opacity_from_confidence",
    }
    defaults = dict(config.get("OPTIMIZATION", {}))
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
        self.eval = False
        # Optional per-image geometry priors. Files are matched by image stem.
        self.depth_prior_dir = ""
        self.normal_prior_dir = ""
        self.confidence_prior_dir = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
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
        self.dense_prior_path = ""
        self.dense_prior_confidence_threshold = 0.3
        self.dense_prior_voxel_size = 0.01
        self.dense_prior_knn_k = 8
        self.initialize_rotation_from_normal = False
        self.initialize_opacity_from_confidence = False
        super().__init__(parser, "Optimization Parameters")

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
