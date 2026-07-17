"""Compare Stage 1 and Stage 1+Stage 2 on a non-training manifest split."""

import argparse
import json
import os
import torch

from stage2_refiner.checkpoint import load_checkpoint
from stage2_refiner.evaluator import evaluate_loader, summarize_by_scene
from stage2_refiner.geometry import tiled_inference
from stage2_refiner.trainer import make_dataset, make_loader
from stage2_refiner.utils import load_config, model_from_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", default="evaluation/stage2"); parser.add_argument("--device")
    args = parser.parse_args(); cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = make_dataset(cfg, args.manifest, args.split); loader = make_loader(dataset, cfg, False)
    model = model_from_config(cfg).to(device); state = load_checkpoint(args.checkpoint, model, map_location=device)
    if state.get("ema") is not None: model.load_state_dict(state["ema"])
    infer = cfg.get("INFERENCE", {})
    tile_fn = lambda m, r, d, n, a: tiled_inference(m, r, d, n, a, int(infer.get("TILE_SIZE", 512)), int(infer.get("TILE_OVERLAP", 32)))
    lpips_model = None
    if cfg.get("EVALUATION", {}).get("LPIPS", True):
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as error:
            print(f"LPIPS unavailable; reporting it as null: {error}")
    summary, rows = evaluate_loader(model, loader, device, args.output_dir, tile_fn, lpips_model)
    report = {"summary": summary, "per_scene": summarize_by_scene(rows), "samples": rows}
    write_json(report, os.path.join(args.output_dir, "metrics.json")); print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__": main()
