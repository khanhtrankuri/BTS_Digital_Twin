from types import SimpleNamespace

import torch

from utils.multiview_rgb import multiview_rgb_loss, warp_source_rgb_to_target
from utils.source_view_selection import select_source_views
from utils.spatial_densification import (
    allocate_tile_budget,
    spatially_balanced_topk,
    tile_error_energy,
)


def _camera(name, center=(0.0, 0.0, 0.0), sharpness=1.0, frame_time=None):
    camera_to_world = torch.eye(4)
    camera_to_world[3, :3] = torch.tensor(center)
    world_view = torch.linalg.inv(camera_to_world)
    return SimpleNamespace(
        image_name=name,
        image_width=9,
        image_height=9,
        fx=8.0,
        fy=8.0,
        cx=4.5,
        cy=4.5,
        world_view_transform=world_view,
        camera_center=torch.tensor(center),
        normalized_sharpness=sharpness,
        mean_brightness=0.5,
        frame_time=frame_time,
        has_ground_truth=True,
    )


def _cfg(**overrides):
    values = dict(
        multiview_position_weight=1.0,
        multiview_angle_weight=1.0,
        multiview_overlap_weight=1.0,
        multiview_sharpness_weight=1.0,
        multiview_temporal_weight=0.0,
        multiview_exposure_mismatch_weight=0.25,
        multiview_min_overlap=0.2,
        multiview_temporal_bracketing=False,
        multiview_sigma_z=0.02,
        multiview_relative_depth_threshold=0.05,
        multiview_min_alpha=0.01,
        multiview_min_valid_pixels=1,
        multiview_patch_size=3,
        multiview_zncc_weight=1.0,
        multiview_gradient_weight=0.2,
        multiview_charbonnier_weight=0.1,
        multiview_charbonnier_eps=1e-3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_selection_excludes_target_and_prefers_sharp_equivalent_view():
    target = _camera("target", frame_time=1.0)
    blurred = _camera("blurred", sharpness=0.1, frame_time=0.0)
    sharp = _camera("sharp", sharpness=1.0, frame_time=2.0)
    selected = select_source_views(target, [target, blurred, sharp], 2, 1.0, _cfg())
    assert [camera.image_name for camera in selected] == ["sharp"]


def test_temporal_bracketing_selects_one_view_on_each_side():
    target = _camera("target", frame_time=10.0)
    candidates = [_camera("before", frame_time=9.0), _camera("after", frame_time=11.0),
                  _camera("far", frame_time=30.0)]
    selected = select_source_views(
        target, candidates, 2, 1.0,
        _cfg(multiview_temporal_bracketing=True, multiview_temporal_weight=1.0))
    assert {camera.image_name for camera in selected} == {"before", "after"}


def test_identity_rgb_reprojection_and_zncc_are_consistent():
    camera = _camera("camera")
    depth = torch.full((1, 9, 9), 2.0, requires_grad=True)
    alpha = torch.ones_like(depth)
    u = torch.linspace(0.0, 1.0, 9)
    source = torch.stack(torch.meshgrid(u, u, indexing="ij") + (torch.ones((9, 9)),), dim=0)
    warped = warp_source_rgb_to_target(
        depth, camera, source, depth.detach(), camera, alpha, alpha, None, _cfg())
    assert warped["valid_mask"].all()
    torch.testing.assert_close(warped["warped_rgb"], source, atol=1e-6, rtol=1e-6)
    losses = multiview_rgb_loss(source, warped["warped_rgb"], warped["valid_mask"],
                                warped["depth_confidence"], _cfg())
    assert losses["total"] < 1e-4
    losses["total"].backward()
    assert depth.grad is not None


def test_visibility_rejects_occluded_source_depth():
    camera = _camera("camera")
    target_depth = torch.full((1, 9, 9), 2.0)
    source_depth = torch.full((1, 9, 9), 1.0)
    source = torch.ones((3, 9, 9))
    warped = warp_source_rgb_to_target(
        target_depth, camera, source, source_depth, camera,
        torch.ones_like(target_depth), torch.ones_like(target_depth), None, _cfg())
    assert not warped["valid_mask"].any()


def test_invalid_reprojection_coordinates_have_finite_zero_gradient():
    camera = _camera("camera")
    target_depth = torch.full((1, 9, 9), 2.0, requires_grad=True)
    with torch.no_grad():
        target_depth[:, 0, 0] = float("nan")
        target_depth[:, 0, 1] = float("inf")
    source_depth = torch.full((1, 9, 9), 2.0)
    source = torch.rand((3, 9, 9))
    alpha = torch.ones_like(source_depth)
    warped = warp_source_rgb_to_target(
        target_depth, camera, source, source_depth, camera,
        alpha, alpha, None, _cfg())
    loss = multiview_rgb_loss(
        source, warped["warped_rgb"], warped["valid_mask"],
        warped["depth_confidence"], _cfg()) ["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(target_depth.grad).all()


def test_flat_zncc_patch_has_finite_rgb_gradient():
    target = torch.full((3, 9, 9), 0.5, requires_grad=True)
    source = torch.full((3, 9, 9), 0.5)
    valid = torch.ones((1, 9, 9), dtype=torch.bool)
    confidence = torch.ones((1, 9, 9))
    loss = multiview_rgb_loss(target, source, valid, confidence, _cfg())["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(target.grad).all()


def test_tile_budget_is_bounded_normalized_and_selection_is_unique():
    residual = torch.zeros((1, 8, 8))
    residual[:, :4, :4] = 4.0
    residual[:, 4:, 4:] = 1.0
    energy = tile_error_energy(residual, torch.zeros_like(residual), tile_size=4)
    budget = allocate_tile_budget(energy, total_budget=12, gamma=0.5, minimum=2, maximum=8)
    assert budget.sum() == 12
    assert budget.max() <= 8
    assert budget[0, 0] > budget[1, 1]

    centers = torch.tensor([[-0.75, -0.75], [-0.25, -0.25], [0.25, 0.25], [0.75, 0.75]])
    selection = spatially_balanced_topk(
        torch.ones(4, dtype=torch.bool), torch.tensor([1.0, 2.0, 3.0, 4.0]),
        centers, energy, 8, 8, total_budget=4, tile_size=4,
        gamma=0.5, minimum=1, maximum=2)
    assert selection.selected.sum() <= 4
    assert selection.tile_selected.sum() == selection.selected.sum()
