import torch

from utils.exposure_utils import PerViewExposure, frame_time_from_name


def test_pose_confidence_blend_falls_back_toward_identity():
    module = PerViewExposure(2)
    module.set_camera_poses([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                            [1000, 1000], [0, 10])
    module.load_gain_bias(torch.full((2, 3), 1.2), torch.full((2, 3), 0.08))
    near_gain, near_bias = module.infer_gain_bias(
        torch.tensor([0.0, 0, 0]), torch.tensor([0, 0, 1]), "pose_confidence_blend")
    far_gain, far_bias = module.infer_gain_bias(
        torch.tensor([100.0, 0, 0]), torch.tensor([0, 0, 1]), "pose_confidence_blend")
    assert torch.linalg.vector_norm(near_gain - 1.0) > torch.linalg.vector_norm(far_gain - 1.0)
    assert torch.linalg.vector_norm(near_bias) > torch.linalg.vector_norm(far_bias)


def test_temporal_mode_selects_interleaved_neighbor():
    module = PerViewExposure(3)
    module.set_camera_poses([[0, 0, 0]] * 3, [[0, 0, 1]] * 3, times=[0, 10, 20])
    module.load_gain_bias(torch.tensor([[0.8] * 3, [1.0] * 3, [1.2] * 3]), torch.zeros(3, 3))
    gain, _ = module.infer_gain_bias(
        torch.zeros(3), torch.tensor([0, 0, 1]), "temporal_weighted", time=19, k=1)
    assert torch.allclose(gain, torch.full((3,), 1.2), atol=1e-5)
    assert frame_time_from_name("DJI_20240101_001234.JPG") == 1234.0

