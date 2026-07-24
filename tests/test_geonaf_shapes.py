import torch

from models.nafnet.geometry_guided_nafnet import GeometryGuidedNAFNet


def test_geonaf_input_output_shapes_and_range():
    model = GeometryGuidedNAFNet(
        in_channels=10,
        width=4,
        enc_blk_nums=(1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1),
    )
    value = torch.rand(2, 10, 17, 19)
    output = model(value)
    assert output["raw_output"].shape == (2, 4, 17, 19)
    assert output["final_rgb"].shape == (2, 3, 17, 19)
    assert output["refine_mask"].shape == (2, 1, 17, 19)
    assert torch.all((0.0 <= output["final_rgb"]) & (output["final_rgb"] <= 1.0))


def test_untrained_geonaf_is_identity():
    model = GeometryGuidedNAFNet(
        in_channels=10,
        width=4,
        enc_blk_nums=(1,),
        middle_blk_num=1,
        dec_blk_nums=(1,),
    )
    value = torch.rand(1, 10, 9, 11)
    assert torch.equal(model(value)["final_rgb"], value[:, :3])
