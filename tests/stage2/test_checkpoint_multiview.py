import torch

from stage2_refiner.checkpoint import ModelEMA, load_checkpoint, save_checkpoint
from stage2_refiner.model import GeometryGuidedNAFNet
from stage2_refiner.multiview import inverse_warp, masked_multiview_l1


def model(): return GeometryGuidedNAFNet(width=8, encoder_blocks=(1,), decoder_blocks=(1,), middle_blocks=1)


def test_checkpoint_restores_model_optimizer_step_and_ema(tmp_path):
    first = model(); optimizer = torch.optim.AdamW(first.parameters()); ema = ModelEMA(first)
    save_checkpoint(tmp_path / "test.pth", first, optimizer=optimizer, ema=ema, epoch=3, step=17)
    second = model(); second_optimizer = torch.optim.AdamW(second.parameters()); second_ema = ModelEMA(second)
    state = load_checkpoint(tmp_path / "test.pth", second, second_optimizer, ema=second_ema)
    assert state["epoch"] == 3 and state["step"] == 17
    for a, b in zip(first.parameters(), second.parameters()): assert torch.equal(a, b)


def test_identity_warp_and_empty_mask_loss():
    image = torch.rand(1, 3, 8, 9); depth = torch.ones(1, 1, 8, 9)
    intrinsics = torch.tensor([[[5., 0., 4.], [0., 5., 3.5], [0., 0., 1.]]]); pose = torch.eye(4).unsqueeze(0)
    warped, valid = inverse_warp(image, depth, intrinsics, intrinsics, pose, pose)
    assert valid.all() and torch.allclose(warped, image, atol=1e-5)
    invalid_pose = pose.clone(); invalid_pose[:, 2, 3] = -10
    _, invalid = inverse_warp(image, depth, intrinsics, intrinsics, invalid_pose, pose)
    assert not invalid.any()
    assert masked_multiview_l1(warped, image, torch.zeros_like(valid)).item() == 0
