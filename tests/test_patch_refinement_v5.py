from types import SimpleNamespace

import torch

from utils.patch_refinement import PatchCrop, crop_camera, sample_patch


def test_patch_sampling_stays_in_bounds_and_ratios_validate():
    residual = torch.zeros((1, 12, 10))
    residual[:, 8:, 6:] = 1.0
    edge = torch.zeros_like(residual)
    crop = sample_patch(residual, edge, 6, random_ratio=0.0, residual_ratio=1.0, edge_ratio=0.0)
    assert crop.category == "residual"
    assert 0 <= crop.left <= 4 and 0 <= crop.top <= 6
    assert crop.width == crop.height == 6


def test_crop_intrinsics_shift_principal_point_for_portrait_image():
    camera = SimpleNamespace(
        fx=600.0, fy=620.0, cx=360.0, cy=640.0,
        znear=0.01, zfar=100.0, FoVx=1.0, FoVy=1.0,
        world_view_transform=torch.eye(4), camera_center=torch.zeros(3),
    )
    cropped = crop_camera(camera, PatchCrop(left=100, top=300, width=512, height=512, category="random"))
    assert cropped.fx == 600.0 and cropped.fy == 620.0
    assert cropped.cx == 260.0 and cropped.cy == 340.0
    assert cropped.image_width == cropped.image_height == 512
    assert torch.isfinite(cropped.full_proj_transform).all()


def test_patch_sampler_uses_thin_mask_and_minimum_distance(monkeypatch):
    residual = torch.zeros((1, 32, 48))
    edge = torch.zeros_like(residual)
    thin = torch.zeros_like(residual)
    thin[:, 20:24, 36:40] = 1.0
    monkeypatch.setattr("utils.patch_refinement.random.random", lambda: 0.95)

    first = sample_patch(residual, edge, 8, thin_structure_map=thin)
    assert first.category == "edge"
    first_center = (first.top + 3.5, first.left + 3.5)
    second = sample_patch(
        residual, edge, 8, thin_structure_map=thin,
        previous_centers=[first_center], min_patch_distance=16.0)
    second_center = (second.top + 3.5, second.left + 3.5)
    distance = ((second_center[0] - first_center[0]) ** 2
                + (second_center[1] - first_center[1]) ** 2) ** 0.5
    assert distance >= 16.0
