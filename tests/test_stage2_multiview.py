import torch

from utils.stage2_multiview import forward_warp_rgb


def test_identity_camera_forward_warp_is_identity_and_masks_dynamic_pixels():
    height, width = 5, 6
    rgb = torch.rand(3, height, width)
    depth = torch.ones(1, height, width)
    alpha = torch.ones_like(depth)
    uncertainty = torch.zeros_like(depth)
    dynamic = torch.zeros_like(depth)
    dynamic[:, 2, 3] = 1.0
    intrinsics = torch.tensor([8.0, 8.0, width / 2.0, height / 2.0])
    extrinsics = torch.eye(4)
    warped, valid = forward_warp_rgb(
        rgb,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        source_alpha=alpha,
        target_alpha=alpha,
        source_uncertainty=uncertainty,
        target_uncertainty=uncertainty,
        dynamic_mask=dynamic,
    )
    expected_valid = torch.ones_like(valid)
    expected_valid[:, 2, 3] = 0.0
    assert torch.equal(valid, expected_valid)
    assert torch.allclose(warped[:, valid[0].bool()], rgb[:, valid[0].bool()])
