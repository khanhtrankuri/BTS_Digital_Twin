import math
from types import SimpleNamespace

import torch

from scene.pose_refinement import PerViewPoseRefinement, se3_exp


def test_se3_exponential_identity_and_known_rotation():
    zero = torch.zeros((1, 3))
    torch.testing.assert_close(se3_exp(zero, zero), torch.eye(4)[None])
    rotation = torch.tensor([[0.0, 0.0, math.pi / 2.0]])
    transform = se3_exp(rotation, zero)[0]
    torch.testing.assert_close(transform[:2, :2], torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
                               atol=1e-5, rtol=1e-5)


def test_pose_bounds_and_identity_initialization():
    model = PerViewPoseRefinement(3, scene_radius=2.0, max_rotation_deg=0.5,
                                  max_translation_radius_ratio=0.005)
    rotation, translation = model.corrections()
    assert torch.count_nonzero(rotation) == 0 and torch.count_nonzero(translation) == 0
    with torch.no_grad():
        model.rotation_raw.fill_(100.0)
        model.translation_raw.fill_(100.0)
    rotation, translation = model.corrections()
    assert torch.linalg.vector_norm(rotation, dim=-1).max() <= math.radians(0.5) + 1e-7
    assert torch.linalg.vector_norm(translation, dim=-1).max() <= 0.01 + 1e-7


def test_refined_camera_preserves_intrinsics_and_has_pose_gradients():
    camera = SimpleNamespace(
        view_index=0,
        world_view_transform=torch.eye(4),
        projection_matrix=torch.eye(4),
        fx=100.0,
    )
    model = PerViewPoseRefinement(1, scene_radius=1.0)
    refined = model.refine_camera(camera)
    assert refined.fx == camera.fx
    torch.testing.assert_close(refined.world_view_transform, torch.eye(4))
    loss = refined.camera_center.square().sum() + refined.full_proj_transform.square().sum()
    loss.backward()
    assert model.rotation_raw.grad is not None and model.translation_raw.grad is not None


def test_pose_state_dict_round_trip_and_smoothness_is_finite():
    first = PerViewPoseRefinement(4, scene_radius=1.0)
    with torch.no_grad():
        first.rotation_raw.normal_()
        first.translation_raw.normal_()
    state = first.state_dict()
    second = PerViewPoseRefinement(4, scene_radius=1.0)
    second.load_state_dict(state)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)
    assert torch.isfinite(second.trajectory_smoothness())

