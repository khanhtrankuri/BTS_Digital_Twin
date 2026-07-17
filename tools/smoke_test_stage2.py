"""Synthetic Stage 2 smoke test: train, checkpoint/resume and tiled inference."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from stage2_refiner.checkpoint import ModelEMA, load_checkpoint, save_checkpoint
from stage2_refiner.geometry import tiled_inference
from stage2_refiner.losses import Stage2Loss
from stage2_refiner.model import GeometryGuidedNAFNet


def make_model(mode, device):
    return GeometryGuidedNAFNet(width=8, encoder_blocks=(1,), decoder_blocks=(1,), middle_blocks=1,
                                geometry_mode=mode).to(device)


def run_batches(mode, steps, device):
    model = make_model(mode, device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = Stage2Loss({"LOSS": {"MSE_WEIGHT": 1.0, "RESIDUAL_WEIGHT": .001}})
    last = None
    for step in range(steps):
        x = torch.rand(1, 8, 16, 18, device=device); target = (x[:, :3] * .9 + .02).clamp(0, 1)
        refined, residual = model(x, return_residual=True)
        losses = criterion(refined, residual, target, x[:, :3], x[:, 3:4], x[:, 4:7], x[:, 7:8], step, steps)
        if not torch.isfinite(losses["total"]): raise FloatingPointError(f"non-finite loss in {mode}")
        optimizer.zero_grad(set_to_none=True); losses["total"].backward(); optimizer.step(); last = losses["total"].item()
    return model, optimizer, last


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device")
    parser.add_argument("--rgb_steps", type=int, default=10); parser.add_argument("--geometry_steps", type=int, default=10)
    parser.add_argument("--full_steps", type=int, default=100); args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); results = {}
    _, _, results["rgb_only_loss"] = run_batches("none", args.rgb_steps, device)
    _, _, results["rgb_geometry_loss"] = run_batches("full", args.geometry_steps, device)
    model, optimizer, results["full_loss"] = run_batches("full", args.full_steps, device)
    ema = ModelEMA(model); ema.update(model)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "smoke.pth"; save_checkpoint(checkpoint, model, optimizer=optimizer, ema=ema, epoch=1, step=args.full_steps)
        restored = make_model("full", device); restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3); restored_ema = ModelEMA(restored)
        state = load_checkpoint(checkpoint, restored, restored_optimizer, ema=restored_ema, map_location=device)
        if state["step"] != args.full_steps: raise AssertionError("checkpoint step was not restored")
        x = torch.rand(1, 8, 33, 35, device=device)
        output, residual = tiled_inference(restored, x[:, :3], x[:, 3:4], x[:, 4:7], x[:, 7:8], tile_size=24, overlap=8)
        if output.shape != (1, 3, 33, 35) or not torch.isfinite(output).all(): raise AssertionError("tiled inference failed")
        # Continue one optimizer step after restore.
        loss = (restored(x) - x[:, :3]).square().mean(); restored_optimizer.zero_grad(); loss.backward(); restored_optimizer.step()
    results.update({"device": str(device), "checkpoint_resume": True, "tiled_shape": list(output.shape), "finite": True})
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
