"""Train the geometry-guided residual refiner from an exported manifest."""

import argparse
import os

from stage2_refiner.trainer import train
from stage2_refiner.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", help="cuda, cuda:0, or cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("DATA", {})["MANIFEST"] = os.path.abspath(args.manifest)
    train(cfg, args.manifest, args.output_dir, args.resume, args.device)


if __name__ == "__main__": main()
