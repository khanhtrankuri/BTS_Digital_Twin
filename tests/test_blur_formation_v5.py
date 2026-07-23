from types import SimpleNamespace

import torch

from scene.blur_model import PerViewBlurTrajectory
from utils.blur_rendering import render_with_blur_formation


def test_blur_subposes_identity_initialization_and_sharp_bypass():
    model = PerViewBlurTrajectory(torch.tensor([True, False]), scene_radius=1.0)
    poses = model.subposes(0, 3)
    torch.testing.assert_close(poses, torch.eye(4)[None].repeat(3, 1, 1))
    assert model.subposes(1, 3).shape[0] == 1


def test_blur_trajectory_bounds_and_regularization():
    model = PerViewBlurTrajectory(torch.tensor([True]), scene_radius=2.0,
                                  max_rotation_deg=0.3,
                                  max_translation_radius_ratio=0.003)
    with torch.no_grad():
        model.rotation_raw.fill_(100.0)
        model.translation_raw.fill_(100.0)
    rotation, translation = model.endpoints()
    assert torch.linalg.vector_norm(rotation, dim=-1).max() <= torch.deg2rad(torch.tensor(0.3)) + 1e-7
    assert torch.linalg.vector_norm(translation, dim=-1).max() <= 0.006 + 1e-7
    assert torch.isfinite(model.regularization())


def test_blur_render_averages_subposes_and_sharp_view_renders_once():
    camera = SimpleNamespace(
        view_index=0, world_view_transform=torch.eye(4), projection_matrix=torch.eye(4))
    model = PerViewBlurTrajectory(torch.tensor([True]), scene_radius=1.0)
    calls = []

    def renderer(current_camera, _gaussians):
        calls.append(current_camera)
        value = current_camera.world_view_transform[3, 0]
        return {
            "render": value.expand(3, 2, 2),
            "canonical_render": value.expand(3, 2, 2),
            "corrected_render": value.expand(3, 2, 2),
            "depth": value.expand(1, 2, 2),
            "normal": None, "alpha": None, "uncertainty": None,
            "depth_variance": None, "foreground_render": None, "background_render": None,
            "radii": torch.tensor([1.0]), "visibility_filter": torch.tensor([True]),
            "viewspace_points": torch.zeros((1, 3)),
        }

    package = render_with_blur_formation(
        camera, object(), model, renderer, SimpleNamespace(blur_num_subposes=3))
    assert len(calls) == 3 and package["blur_num_subposes"] == 3
    assert torch.isfinite(package["render"]).all()

