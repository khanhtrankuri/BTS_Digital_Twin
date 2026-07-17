import torch

from stage2_refiner.losses import Stage2Loss
from stage2_refiner.model import GeometryGuidedNAFNet


def test_ten_batch_training_smoke():
    model = GeometryGuidedNAFNet(width=8, encoder_blocks=(1,), decoder_blocks=(1,), middle_blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = {"LOSS": {"MSE_WEIGHT": 1.0}}
    criterion = Stage2Loss(cfg)
    for step in range(10):
        x = torch.rand(1, 8, 16, 16); target = x[:, :3] * .9
        refined, residual = model(x, return_residual=True)
        losses = criterion(refined, residual, target, x[:, :3], x[:, 3:4], x[:, 4:7], x[:, 7:8], step, 10)
        assert torch.isfinite(losses["total"])
        optimizer.zero_grad(); losses["total"].backward(); optimizer.step()
