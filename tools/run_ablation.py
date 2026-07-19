"""Reproducible BTS-GeoGS-v4 ablation runner.

Stage-1 experiments A0--A10 are executable. A11--A16 are deliberately gated
until the shared multiview Stage-2 baseline has been validated; requesting one
fails rather than silently running an incomparable model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import deep_merge, load_yaml_with_base


STAGE1_EXPERIMENTS = {f"A{index}" for index in range(11)}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: _version(name) for name in
                     ("torch", "torchvision", "numpy", "opencv-python", "PyYAML", "lpips")},
    }


def _git_hash() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _write_yaml(path: Path, config: dict) -> None:
    import yaml
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run(options: argparse.Namespace) -> Path:
    invalid = set(options.experiments) - STAGE1_EXPERIMENTS
    if invalid:
        raise RuntimeError(
            f"Experiments {sorted(invalid)} are gated: Phase 3/4 must not run before strict validation "
            "and the shared multiview interface are verified.")
    data_root = options.data_root or os.environ.get("BTS_DATA_ROOT")
    if not data_root:
        raise ValueError("Provide --data_root or set BTS_DATA_ROOT")
    data_root = Path(data_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = options.output_root.resolve() / timestamp
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for experiment in options.experiments:
        overlay_path = REPO_ROOT / "configs" / "bts_v4" / "ablations" / f"{experiment}.yaml"
        overlay = load_yaml_with_base(overlay_path)
        for scene in options.scenes:
            scene_config = load_yaml_with_base(REPO_ROOT / "configs" / "bts_v4" / f"{scene}.yaml")
            resolved = deep_merge(scene_config, overlay)
            undistorted = bool(resolved.get("CAMERA", {}).get("USE_UNDISTORTED_DATA", False))
            if undistorted:
                if options.undistorted_root is None:
                    raise ValueError(f"{experiment}/{scene} requires --undistorted_root")
                source = options.undistorted_root.resolve() / scene
            else:
                source = data_root / scene
            if not source.exists():
                raise FileNotFoundError(source)
            for seed in options.seeds:
                run_dir = root / experiment / scene / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                config_path = run_dir / "resolved_config.yaml"
                _write_yaml(config_path, resolved)
                model_path = run_dir / "model"
                command = [sys.executable, str(REPO_ROOT / "train.py"), "-s", str(source),
                           "-m", str(model_path), "--config", str(config_path),
                           "--seed", str(seed), "--disable_viewer"]
                manifest = {
                    "experiment": experiment, "scene": scene, "seed": seed,
                    "source": str(source), "model_path": str(model_path),
                    "git_commit": _git_hash(), "command": command,
                    "environment": _environment(), "status": "planned" if options.dry_run else "running",
                }
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                if not options.dry_run:
                    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT, text=True)
                    manifest["status"] = "complete" if completed.returncode == 0 else "failed"
                    manifest["return_code"] = completed.returncode
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    if completed.returncode != 0 and not options.continue_on_error:
                        raise RuntimeError(f"Training failed; inspect {run_dir / 'train.log'}")
                rows.append({"experiment": experiment, "scene": scene, "seed": seed,
                             "status": manifest["status"], "run_dir": str(run_dir)})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["experiment", "scene", "seed", "status", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--undistorted_root", type=Path)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "output" / "ablations")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser


if __name__ == "__main__":
    output = run(build_parser().parse_args())
    print(f"Ablation outputs: {output}")
