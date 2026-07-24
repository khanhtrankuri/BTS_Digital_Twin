import torch
from torch import nn

from utils.stage2_inference import tiled_refine


class PointwiseRefiner(nn.Module):
    def forward(self, value):
        gaussian = value[:, :3]
        delta = torch.tanh(value[:, 3:6])
        mask = torch.sigmoid(value[:, 6:7])
        correction = 0.15 * mask * delta
        return {
            "final_rgb": (gaussian + correction).clamp(0, 1),
            "delta_rgb": delta,
            "refine_mask": mask,
            "effective_mask": mask,
            "correction": correction,
        }


def test_tiled_output_matches_full_without_seams():
    torch.manual_seed(3)
    value = torch.rand(1, 10, 37, 43)
    model = PointwiseRefiner()
    full = model(value)
    tiled = tiled_refine(model, value, tile_size=16, overlap=6)
    for key in full:
        assert torch.allclose(tiled[key], full[key], atol=1e-6)
