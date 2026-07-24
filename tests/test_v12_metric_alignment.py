from argparse import ArgumentParser
import random

import torch

from arguments import ModelParams, OptimizationParams, load_bts_geogs_config
from utils.perceptual_utils import native_resolution_crops
from utils.training_schedules import get_resolution_stage, get_stage_loss_weights


def test_native_perceptual_crop_preserves_pixels_targets_error_and_backpropagates():
    image = torch.zeros((3, 8, 8), requires_grad=True)
    target = torch.zeros_like(image)
    target[:, 4:, 4:] = 1.0
    crops, target_crops = native_resolution_crops(
        image, target, crop_size=4, num_crops=1)
    assert crops.shape == (1, 3, 4, 4)
    assert target_crops.mean().item() == 1.0
    crops.sum().backward()
    assert image.grad is not None
    assert image.grad[:, 4:, 4:].sum().item() == 48.0


def test_native_perceptual_random_crops_are_seed_reproducible():
    image = torch.rand((3, 12, 10))
    target = image.clone()
    random.seed(7)
    first = native_resolution_crops(image, target, crop_size=6, num_crops=3)[0]
    random.seed(7)
    second = native_resolution_crops(image, target, crop_size=6, num_crops=3)[0]
    torch.testing.assert_close(first, second)


def test_v12_quality_config_is_native_resolution_and_metric_aligned():
    config = load_bts_geogs_config("configs/bts_v12/quality.yaml")
    assert config["iterations"] == 20_000
    assert config["resolution"] == 1
    assert config["resolution_schedule_enabled"] is True
    assert config["validation_full_resolution"] is True
    assert config["densify_until_iter"] == 2500
    assert config["perceptual_loss_enabled"] is True
    assert config["perceptual_loss_mode"] == "native_crops"
    assert config["multiscale_ssim_enabled"] is True
    assert get_resolution_stage(0, type("Cfg", (), config)) == (0, 0.5)
    assert get_resolution_stage(6000, type("Cfg", (), config)) == (2, 1.0)


def test_v12_late_stage_prioritizes_structure_and_stabilizes_geometry():
    config = load_bts_geogs_config("configs/bts_v12/quality.yaml")
    cfg = type("Cfg", (), config)
    weights = get_stage_loss_weights(9000, cfg)
    assert weights["dssim"] == 0.45
    assert weights["l1"] + weights["mse"] + weights["dssim"] == 1.0
    assert config["lr_stage_c_xyz"] == 0.10
    assert config["lr_stage_c_features"] == 0.75


def test_v12_8gb_ablation_configs_have_matched_resource_budgets():
    control = load_bts_geogs_config("configs/bts_v12/control_v11_8gb.yaml")
    fullres = load_bts_geogs_config("configs/bts_v12/fullres_8gb.yaml")
    quality = load_bts_geogs_config("configs/bts_v12/quality_8gb.yaml")
    for key in (
        "iterations",
        "resolution",
        "resolution_schedule_stages",
        "max_gaussians",
        "max_new_gaussians_per_step",
    ):
        assert fullres[key] == quality[key]
    assert control["max_gaussians"] == fullres["max_gaussians"]
    assert (
        control["max_new_gaussians_per_step"]
        == fullres["max_new_gaussians_per_step"]
    )
    assert fullres.get("perceptual_loss_enabled", False) is False
    assert quality["perceptual_loss_enabled"] is True


def test_v12_hcm_8gb_controls_have_matched_caps():
    control = load_bts_geogs_config("configs/bts_v12/control_v11_hcm_8gb.yaml")
    fullres = load_bts_geogs_config("configs/bts_v12/fullres_hcm_8gb.yaml")
    assert control["max_gaussians"] == fullres["max_gaussians"] == 350_000
    assert (
        control["max_new_gaussians_per_step"]
        == fullres["max_new_gaussians_per_step"]
        == 15_000
    )


def test_validation_full_resolution_is_available_on_model_params_cli():
    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    args = parser.parse_args(["--validation_full_resolution"])
    assert args.validation_full_resolution is True
