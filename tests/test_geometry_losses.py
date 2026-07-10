import torch

from utils.geometry_losses import (
    edge_weighted_l1_loss,
    gaussian_scale_regularization,
    normal_consistency_loss,
    scale_shift_invariant_depth_loss,
)


def test_depth_is_zero_for_equal_and_scale_shift_aligned_inputs():
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    assert scale_shift_invariant_depth_loss(target, target).item() < 1e-6
    assert scale_shift_invariant_depth_loss(target * 2.0 + 3.0, target).item() < 1e-5


def test_normal_abs_cosine_handles_flipped_normals():
    normal = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])
    assert normal_consistency_loss(normal, -normal, use_abs_cosine=True).item() < 1e-6


def test_edge_weighted_loss_emphasizes_errors_on_edges():
    target = torch.zeros(3, 2, 2)
    pred = target.clone()
    pred[:, 0, 0] = 1.0
    edge = torch.zeros(1, 2, 2)
    edge[:, 0, 0] = 1.0
    assert edge_weighted_l1_loss(pred, target, edge, gamma=2.0) > edge_weighted_l1_loss(pred, target, torch.zeros_like(edge))


def test_scale_regularization_only_penalizes_violations():
    assert gaussian_scale_regularization(torch.full((2, 3), 0.05), 0.1, 20.0).item() == 0.0
    assert gaussian_scale_regularization(torch.tensor([[0.2, 0.05, 0.05]]), 0.1, 20.0).item() > 0.0
