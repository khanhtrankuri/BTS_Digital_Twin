import torch

from models.nafnet.geometry_guided_nafnet import GeometryGuidedNAFNet


def test_phase2_detached_input_does_not_backpropagate_to_gaussian():
    gaussian_parameter = torch.nn.Parameter(torch.rand(1, 10, 8, 8))
    exported_or_online_detached = gaussian_parameter.detach()
    model = GeometryGuidedNAFNet(
        in_channels=10,
        width=4,
        enc_blk_nums=(1,),
        middle_blk_num=1,
        dec_blk_nums=(1,),
    )
    loss = model(exported_or_online_detached)["final_rgb"].mean()
    loss.backward()
    assert gaussian_parameter.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
