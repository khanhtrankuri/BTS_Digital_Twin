from types import SimpleNamespace

import torch

from utils.depth_reprojection import pairwise_depth_consistency
from utils.persistent_densification import (bitset_popcount, gradient_burstiness,
                                            recent_hit_count, update_persistent_window,
                                            update_view_support)
from utils.training_schedules import get_resolution_stage, get_sh_degree
from arguments import load_bts_geogs_config
from utils.sky_utils import DirectionalSHBackground


def test_progressive_schedules_transition_at_exact_milestones():
    cfg = SimpleNamespace(
        sh_schedule_enabled=True, sh_schedule_milestones=[[0, 0], [3, 1], [8, 2]],
        resolution_schedule_enabled=True,
        resolution_schedule_stages=[{"START_ITER": 0, "SCALE": 0.5},
                                    {"START_ITER": 5, "SCALE": 1.0}],
    )
    assert get_sh_degree(7, cfg, 3) == 1
    assert get_sh_degree(8, cfg, 3) == 2
    assert get_resolution_stage(4, cfg) == (0, 0.5)
    assert get_resolution_stage(5, cfg) == (1, 1.0)


def test_view_bitset_and_persistent_recent_hits():
    support = torch.zeros(3, 1, dtype=torch.int64)
    visible = torch.tensor([True, False, True])
    directions = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    update_view_support(support, directions, visible, 12)
    update_view_support(support, torch.tensor([[0.0, 1, 0], [0, 1.0, 0], [1, 0, 0.0]]), visible, 12)
    assert torch.equal(bitset_popcount(support, 12), torch.tensor([2, 0, 2]))

    score_ema = torch.zeros(3, 1)
    hit_ema = torch.zeros(3, 1)
    hit_count = torch.zeros(3, 1, dtype=torch.int64)
    recent = torch.zeros(3, 1, dtype=torch.int64)
    hits = torch.tensor([True, False, True])
    for _ in range(3):
        update_persistent_window(score_ema, hit_ema, hit_count, recent, hits,
                                 torch.ones(3), 0.8, 4)
    assert torch.equal(recent_hit_count(recent, 4), torch.tensor([3, 0, 3]))
    assert gradient_burstiness(torch.tensor([1.0]), torch.tensor([1.0])).item() == 0.0


def test_identity_camera_depth_reprojection_is_consistent():
    camera = SimpleNamespace(fx=100.0, fy=100.0, cx=1.5, cy=1.5,
                             image_width=3, image_height=3,
                             world_view_transform=torch.eye(4))
    depth = torch.full((1, 3, 3), 2.0)
    alpha = torch.ones_like(depth)
    result = pairwise_depth_consistency(depth, depth, camera, camera,
                                        target_alpha=alpha, source_alpha=alpha)
    assert result.hard_visibility.all()
    assert result.relative_error.max() < 1e-5


def test_release_config_resolves_absgrad_inheritance():
    config = load_bts_geogs_config(
        "configs/bts_v11/absgrad_early_stop_15k.yaml"
    )
    assert config["resolution"] == 2
    assert config["iterations"] == 15000
    assert config["densification_method"] == "absolute_gradient"
    assert config["densify_until_iter"] == 2500
    assert config["antialiasing"] is True


def test_directional_background_has_bounded_image_output():
    camera = SimpleNamespace(fx=10.0, fy=10.0, cx=2.0, cy=1.5,
                             image_width=4, image_height=3,
                             world_view_transform=torch.eye(4))
    model = DirectionalSHBackground(2, (0.2, 0.4, 0.6))
    image = model(camera)
    assert image.shape == (3, 3, 4)
    assert torch.all((image > 0.0) & (image < 1.0))
