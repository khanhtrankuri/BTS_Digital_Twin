import torch

from utils.exposure_utils import PerViewExposure


def test_identity_gain_and_zero_bias_preserve_image():
    module = PerViewExposure(2, min_gain=0.75, max_gain=1.25, max_bias=0.10)
    image = torch.rand(3, 4, 5)
    assert torch.equal(module(image, 0), image)
    assert module.regularization_loss().abs().item() < 1e-12


def test_gain_and_bias_are_always_bounded():
    module = PerViewExposure(2, min_gain=0.75, max_gain=1.25, max_bias=0.10)
    with torch.no_grad():
        module.raw_gain.copy_(torch.tensor([[-100.0, 0.0, 100.0], [100.0, -100.0, 0.0]]))
        module.raw_bias.copy_(torch.tensor([[-100.0, 0.0, 100.0], [100.0, -100.0, 0.0]]))
    assert module.gains().min() >= 0.75
    assert module.gains().max() <= 1.25
    assert module.biases().abs().max() <= 0.10


def test_photometric_gradient_reaches_exposure_parameters():
    module = PerViewExposure(1)
    loss = module(torch.ones(3, 2, 2), 0).square().mean()
    loss.backward()
    assert module.raw_gain.grad is not None and module.raw_gain.grad.abs().sum() > 0
    assert module.raw_bias.grad is not None and module.raw_bias.grad.abs().sum() > 0


def test_exposure_state_dict_round_trip(tmp_path):
    source = PerViewExposure(2)
    with torch.no_grad():
        source.raw_gain.add_(0.2)
        source.raw_bias.sub_(0.1)
    path = tmp_path / "exposure.pt"
    torch.save(source.state_dict(), path)
    target = PerViewExposure(2)
    target.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(source.raw_gain, target.raw_gain)
    assert torch.equal(source.raw_bias, target.raw_bias)


def test_nearest_inference_uses_only_camera_pose():
    module = PerViewExposure(2)
    module.set_camera_poses([[0, 0, 0], [10, 0, 0]], [[0, 0, 1], [0, 0, 1]])
    module.load_gain_bias(torch.tensor([[0.9, 0.9, 0.9], [1.1, 1.1, 1.1]]), torch.zeros(2, 3))
    gain, bias = module.infer_gain_bias(torch.tensor([9.9, 0, 0]), torch.tensor([0, 0, 1]), "nearest_camera")
    assert torch.allclose(gain, torch.full((3,), 1.1), atol=1e-5)
    assert torch.equal(bias, torch.zeros(3))
