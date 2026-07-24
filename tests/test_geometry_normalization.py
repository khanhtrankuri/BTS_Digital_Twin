import torch

from utils.stage2_geometry import robust_normalize_geometry


def test_geometry_normalization_is_finite():
    depth = torch.tensor([[[1.0, 2.0], [3.0, float("nan")]]])
    alpha = torch.ones_like(depth)
    variance = torch.tensor([[[0.1, 0.2], [float("inf"), 0.0]]])
    normalized_depth, normalized_variance = robust_normalize_geometry(
        depth, alpha, variance
    )
    assert torch.isfinite(normalized_depth).all()
    assert torch.isfinite(normalized_variance).all()
    assert normalized_depth.abs().max() <= 1.0
    assert normalized_variance.min() >= 0.0
    assert normalized_variance.max() <= 1.0


def test_empty_valid_depth_falls_back_to_zero():
    depth = torch.zeros(1, 4, 5)
    alpha = torch.zeros_like(depth)
    variance = torch.ones_like(depth)
    normalized_depth, normalized_variance = robust_normalize_geometry(
        depth, alpha, variance
    )
    assert torch.count_nonzero(normalized_depth) == 0
    assert torch.count_nonzero(normalized_variance) == 0


def test_constant_depth_is_stable():
    depth = torch.full((1, 4, 5), 3.0)
    alpha = torch.ones_like(depth)
    variance = torch.zeros_like(depth)
    normalized_depth, normalized_variance = robust_normalize_geometry(
        depth, alpha, variance
    )
    assert torch.equal(normalized_depth, torch.zeros_like(depth))
    assert torch.equal(normalized_variance, torch.zeros_like(depth))
