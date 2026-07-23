from types import SimpleNamespace

import torch

from arguments import load_bts_geogs_config
from utils.densification_utils import split_opacity_conserving
from utils.footprint_sampling import project_gaussian_ellipse, sample_footprint_statistics
from utils.geometry_losses import ray_depth_variance_loss
from utils.structure_aligned_split import classify_gaussian_shape, structure_aligned_split
from utils.training_schedules import piecewise_peak_weight


def _split_cfg(**overrides):
    values = dict(
        wire_ratio_threshold=4.0,
        surface_ratio_threshold=4.0,
        wire_split_offset=0.35,
        wire_major_scale_factor=0.58,
        wire_minor_scale_factor=0.85,
        surface_split_offset=0.30,
        blob_random_split=False,
        opacity_conserving_split=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _camera(width=100, height=80):
    return SimpleNamespace(
        image_width=width,
        image_height=height,
        fx=50.0,
        fy=40.0,
        cx=50.0,
        cy=40.0,
        world_view_transform=torch.eye(4),
    )


def test_opacity_conserves_transmittance_and_is_finite_at_extremes():
    parent_raw = torch.tensor([[-30.0], [-10.0], [0.0], [10.0], [30.0]], dtype=torch.float64)
    for outputs in (1, 2, 3, 8):
        child_raw = split_opacity_conserving(parent_raw, outputs)
        assert torch.isfinite(child_raw).all()
        parent_transmittance = 1.0 - torch.sigmoid(parent_raw)
        child_transmittance = (1.0 - torch.sigmoid(child_raw)).pow(outputs)
        torch.testing.assert_close(child_transmittance, parent_transmittance, atol=1e-12, rtol=1e-8)

    saturated = split_opacity_conserving(torch.tensor([[-1000.0], [1000.0]]), 2)
    assert torch.isfinite(saturated).all()


def test_wire_surface_blob_classification_is_disjoint():
    scales = torch.tensor([[8.0, 1.0, 1.0], [4.0, 4.0, 0.5], [1.0, 1.0, 1.0]])
    wire, surface, blob = classify_gaussian_shape(scales, 4.0, 4.0)
    assert wire.tolist() == [True, False, False]
    assert surface.tolist() == [False, True, False]
    assert blob.tolist() == [False, False, True]
    assert torch.stack((wire, surface, blob)).sum(dim=0).eq(1).all()


def test_structure_split_stays_on_wire_and_surface_tangent_plane():
    xyz = torch.zeros((2, 3))
    scales = torch.tensor([[8.0, 1.0, 1.0], [4.0, 4.0, 0.5]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    opacity = torch.zeros((2, 1))
    # Surface error direction points along its normal and must be rejected.
    error_direction = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = structure_aligned_split(
        xyz, scales, rotations, opacity, error_direction, _split_cfg(), num_outputs=2)
    offsets = result.xyz.reshape(2, 2, 3)
    assert torch.allclose(offsets[:, 0, 1:], torch.zeros_like(offsets[:, 0, 1:]))
    assert torch.allclose(offsets[:, 1, 2], torch.zeros_like(offsets[:, 1, 2]))
    assert torch.allclose(offsets[0], -offsets[1])
    assert result.scaling[0, 0] == scales[0, 0] * 0.58
    assert torch.isfinite(result.raw_opacity).all()


def test_ray_depth_variance_loss_matches_relative_variance_and_masks_invalid():
    depth = torch.full((1, 2, 2), 2.0, requires_grad=True)
    variance = torch.full((1, 2, 2), 0.04, requires_grad=True)
    alpha = torch.ones_like(depth)
    loss = ray_depth_variance_loss(depth, variance, alpha)
    torch.testing.assert_close(loss, torch.tensor(0.01), atol=1e-6, rtol=1e-6)
    loss.backward()
    assert variance.grad is not None and torch.isfinite(variance.grad).all()

    zero = ray_depth_variance_loss(depth, variance, alpha, valid_mask=torch.zeros_like(depth, dtype=torch.bool))
    assert zero.requires_grad and zero.item() == 0.0


def test_ellipse_projection_preserves_anisotropy_and_principal_point():
    covariance = torch.diag(torch.tensor([0.04, 0.01, 0.01]))[None]
    centers, major, minor = project_gaussian_ellipse(
        torch.tensor([[0.0, 0.0, 2.0]]), covariance, _camera())
    torch.testing.assert_close(centers, torch.zeros_like(centers), atol=1e-6, rtol=0)
    assert torch.linalg.vector_norm(major) > torch.linalg.vector_norm(minor)
    torch.testing.assert_close(torch.linalg.vector_norm(major), torch.tensor(0.2), atol=1e-5, rtol=1e-5)


def test_footprint_sampling_uses_all_pattern_points_without_border_zero_bias():
    residual = torch.zeros((1, 5, 5))
    residual[0, 2, 2] = 1.0
    edge = residual.clone()
    centers = torch.tensor([[0.0, 0.0], [0.99, 0.99]])
    axes = torch.zeros_like(centers)
    stats = sample_footprint_statistics(residual, edge, centers, axes, axes)
    torch.testing.assert_close(stats["residual_score"][0], torch.tensor(1.0))
    assert stats["valid_sample_count"].tolist() == [9, 9]
    assert torch.isfinite(stats["residual_score"]).all()


def test_ray_variance_piecewise_schedule_and_release_config_inheritance():
    assert piecewise_peak_weight(3999, 4000, 16000, 35000, 0.001, 0.01, 0.002) == 0.0
    assert piecewise_peak_weight(4000, 4000, 16000, 35000, 0.001, 0.01, 0.002) == 0.001
    assert piecewise_peak_weight(16000, 4000, 16000, 35000, 0.001, 0.01, 0.002) == 0.01
    assert piecewise_peak_weight(35000, 4000, 16000, 35000, 0.001, 0.01, 0.002) == 0.002
    assert piecewise_peak_weight(35001, 4000, 16000, 35000, 0.001, 0.01, 0.002) == 0.0

    config = load_bts_geogs_config(
        "configs/bts_v11/absgrad_early_stop_15k.yaml"
    )
    assert config["iterations"] == 15000
    assert config["densification_method"] == "absolute_gradient"
    assert config["densification_abs_grad_threshold"] == 0.0008
    assert config["max_gaussians"] == 2_500_000
