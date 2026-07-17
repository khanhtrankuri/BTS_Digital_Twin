import torch

from stage2_refiner.geometry import tiled_inference
from stage2_refiner.model import GeometryGuidedNAFNet


def tiny_model(**kwargs):
    return GeometryGuidedNAFNet(width=8, encoder_blocks=(1, 1), decoder_blocks=(1, 1), middle_blocks=1, **kwargs)


def test_model_shape_bounds_identity_and_backward():
    model = tiny_model(max_residual=0.15)
    x = torch.rand(2, 8, 31, 35, requires_grad=True)
    refined, residual = model(x, return_residual=True)
    assert refined.shape == residual.shape == (2, 3, 31, 35)
    assert residual.abs().max() <= 0.15 + 1e-6
    assert torch.allclose(refined, x[:, :3], atol=1e-6)
    (refined.mean() + residual.mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_confidence_gate_shape():
    model = tiny_model(confidence_gate=True)
    refined, residual, gate = model(torch.rand(1, 8, 17, 19), return_residual=True, return_gate=True)
    assert refined.shape == residual.shape == (1, 3, 17, 19)
    assert gate.shape == (1, 1, 17, 19)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_tiled_matches_full_for_identity_initialized_model():
    model = tiny_model().eval(); x = torch.rand(1, 8, 41, 45)
    full, _ = model(x, return_residual=True)
    tiled, _ = tiled_inference(model, x[:, :3], x[:, 3:4], x[:, 4:7], x[:, 7:8], tile_size=24, overlap=8)
    assert torch.allclose(full, tiled, atol=1e-5)
