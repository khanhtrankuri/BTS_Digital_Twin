import torch
from torch import nn

from models.nafnet.geometry_guided_nafnet import GeometryGuidedNAFNet


class FixedBackbone(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, value):
        return self.output.expand(value.shape[0], -1, value.shape[2], value.shape[3])


def _model(raw_output, scale=0.15):
    model = GeometryGuidedNAFNet(
        in_channels=10,
        width=4,
        enc_blk_nums=(1,),
        middle_blk_num=1,
        dec_blk_nums=(1,),
        residual_scale=scale,
    )
    model.backbone = FixedBackbone(torch.tensor(raw_output).view(1, 4, 1, 1))
    return model


def test_zero_delta_keeps_gaussian_rgb():
    value = torch.rand(1, 10, 4, 5)
    output = _model([0.0, 0.0, 0.0, 10.0])(value)
    assert torch.equal(output["final_rgb"], value[:, :3])


def test_zero_mask_keeps_gaussian_rgb():
    value = torch.full((1, 10, 4, 5), 0.5)
    output = _model([10.0, 10.0, 10.0, -100.0])(value)
    assert torch.allclose(output["final_rgb"], value[:, :3], atol=1e-7)


def test_correction_never_exceeds_residual_scale():
    value = torch.full((1, 10, 4, 5), 0.5)
    scale = 0.15
    output = _model([100.0, -100.0, 100.0, 100.0], scale)(value)
    assert output["correction"].abs().max() <= scale + 1e-7
