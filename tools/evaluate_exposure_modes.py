"""Render and archive a fair validation ablation of exposure test modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate(options: argparse.Namespace) -> dict:
    model = options.model_path.resolve()
    output = model / "exposure_ablation"
    output.mkdir(exist_ok=True)
    summary = {}
    for mode in options.modes:
        subprocess.run([sys.executable, str(REPO_ROOT / "render.py"), "-m", str(model),
                        "--skip_train", "--exposure_compensation",
                        "--test_exposure_mode", mode], cwd=REPO_ROOT, check=True)
        subprocess.run([sys.executable, str(REPO_ROOT / "metrics.py"), "-m", str(model)],
                       cwd=REPO_ROOT, check=True)
        mode_dir = output / mode
        mode_dir.mkdir(exist_ok=False)
        for name in ("results.json", "per_view.json"):
            source = model / name
            if source.exists():
                shutil.copy2(source, mode_dir / name)
        result_path = mode_dir / "results.json"
        summary[mode] = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=("validation",))
    parser.add_argument("--modes", nargs="+", default=["identity", "nearest_camera", "weighted_nearest",
                                                        "pose_confidence_blend"])
    print(json.dumps(evaluate(parser.parse_args()), indent=2))
