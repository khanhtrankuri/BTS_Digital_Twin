import torch

from utils.densification_utils import (
    accumulate_visible_statistics,
    conservative_opacity_prune_mask,
    corrected_residual_map,
    limit_mask,
    percentile_mask,
    robust_normalize_score,
)


def test_opposing_subgradients_cancel_signed_but_not_absolute_sum():
    subgradients = torch.tensor([[2.0, -1.0], [-2.0, 1.0]])
    assert torch.linalg.vector_norm(subgradients.sum(0)).item() == 0.0
    assert torch.linalg.vector_norm(subgradients.abs().sum(0)).item() > 0.0


def test_robust_normalization_handles_zero_iqr_without_nan():
    values = torch.ones(8)
    normalized = robust_normalize_score(values, torch.ones(8, dtype=torch.bool))
    assert torch.equal(normalized, torch.zeros_like(values))
    assert torch.isfinite(normalized).all()


def test_percentile_and_topk_selection_are_bounded():
    scores = torch.arange(10, dtype=torch.float32)
    selected = percentile_mask(scores, torch.ones(10, dtype=torch.bool), 0.8)
    limited = limit_mask(selected, scores, 1)
    assert selected.sum() == 2
    assert limited.sum() == 1 and limited[-1]


def test_statistics_update_visible_gaussians_only():
    accum = torch.zeros(4, 1)
    denom = torch.zeros(4, 1)
    values = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    visible = torch.tensor([False, True, False, True])
    accumulate_visible_statistics(accum, denom, values, visible)
    assert torch.equal(accum.squeeze(1), torch.tensor([0.0, 1.0, 0.0, 3.0]))
    assert torch.equal(denom.squeeze(1), visible.float())


def test_exposure_correction_removes_brightness_only_residual():
    canonical = torch.full((3, 2, 2), 0.4)
    gt = canonical * 1.2 + 0.05
    canonical_residual = corrected_residual_map(canonical, gt, "l1").mean()
    corrected_residual = corrected_residual_map(canonical * 1.2 + 0.05, gt, "l1").mean()
    assert corrected_residual < canonical_residual


def test_low_opacity_is_not_a_sufficient_pruning_signal():
    opacity = torch.tensor([0.001, 0.001, 0.5, 0.001])
    visibility = torch.tensor([10, 0, 0, 0])
    age = torch.tensor([1000, 1000, 1000, 10])
    selected = conservative_opacity_prune_mask(opacity, visibility, age, 0.005, 200)
    assert selected.tolist() == [False, True, False, False]
