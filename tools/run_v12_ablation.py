"""Run native-resolution BTS v12 ablations and aggregate leaderboard score.

Prepare folds first with ``tools/prepare_v12_validation.py``. This runner keeps
the v11 control, full-resolution change, metric-aligned losses, and optional
multi-view regularizer separate so a costly seven-scene run has a defensible
keep/rollback decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import load_bts_geogs_config
from tools.prepare_v12_validation import SCENES
from tools.train_render_submission_v11 import run_logged


EXPERIMENTS = {
    "control_v11": REPO_ROOT / "configs" / "bts_v12" / "control_v11.yaml",
    "control_v11_8gb": REPO_ROOT / "configs" / "bts_v12" / "control_v11_8gb.yaml",
    "control_v11_700k": REPO_ROOT / "configs" / "bts_v12" / "control_v11_8gb.yaml",
    "control_v11_450k": REPO_ROOT / "configs" / "bts_v12" / "control_v11_8gb.yaml",
    "control_v11_hcm_8gb": REPO_ROOT / "configs" / "bts_v12" / "control_v11_hcm_8gb.yaml",
    "fullres": REPO_ROOT / "configs" / "bts_v12" / "fullres.yaml",
    "fullres_8gb": REPO_ROOT / "configs" / "bts_v12" / "fullres_8gb.yaml",
    "fullres_hcm_8gb": REPO_ROOT / "configs" / "bts_v12" / "fullres_hcm_8gb.yaml",
    "quality": REPO_ROOT / "configs" / "bts_v12" / "quality.yaml",
    "quality_8gb": REPO_ROOT / "configs" / "bts_v12" / "quality_8gb.yaml",
    "multiview": REPO_ROOT / "configs" / "bts_v12" / "quality_multiview.yaml",
}


def _source_for(scene: str, data_root: Path, prepared_root: Path) -> Path:
    return (prepared_root if scene.startswith("HCM") else data_root) / scene


def _latest_checkpoint(model_path: Path, final_iteration: int) -> Path | None:
    candidates = []
    for path in model_path.glob("chkpnt*.pth"):
        match = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
        if match and int(match.group(1)) < final_iteration:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def _training_command(
    python: str,
    config: Path,
    source: Path,
    model_path: Path,
    split_path: Path,
    final_iteration: int,
    seed: int,
    checkpoint: Path | None,
) -> list[str]:
    checkpoint_iterations = sorted({
        value for value in (2000, 7000, 12000, final_iteration)
        if value <= final_iteration
    })
    command = [
        python,
        str(REPO_ROOT / "train.py"),
        "--config",
        str(config),
        "-s",
        str(source),
        "-m",
        str(model_path),
        "--iterations",
        str(final_iteration),
        "--eval",
        "--validation_split_file",
        str(split_path),
        "--validation_full_resolution",
        "--cache_images_on_cpu",
        "--disable_viewer",
        "--test_iterations",
        str(final_iteration),
        "--save_iterations",
        str(final_iteration),
        "--checkpoint_iterations",
        *(str(value) for value in checkpoint_iterations),
        "--psnr_max",
        "50",
        "--seed",
        str(seed),
        "--quiet",
    ]
    if checkpoint is not None:
        command.extend(["--start_checkpoint", str(checkpoint)])
    return command


def _read_summary(model_path: Path) -> dict | None:
    path = model_path / "stage1_validation_metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("splits", {}).get("test", {}).get("summary")


def _aggregate(records: list[dict]) -> dict:
    metric_names = ("score", "psnr", "ssim", "lpips")
    experiments = {}
    for experiment in sorted({record["experiment"] for record in records}):
        selected = [
            record for record in records
            if record["experiment"] == experiment and record.get("summary")
        ]
        experiments[experiment] = {
            "completed_runs": len(selected),
            **{
                metric: (
                    sum(record["summary"][metric] for record in selected)
                    / len(selected)
                    if selected else None
                )
                for metric in metric_names
            },
        }
    control_score = experiments.get("control_v11", {}).get("score")
    if control_score is not None:
        for values in experiments.values():
            if values["score"] is not None:
                values["score_delta_vs_v11"] = values["score"] - control_score
    return {"records": records, "experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Lenovo\Documents\Val_Race"),
    )
    parser.add_argument(
        "--prepared_root",
        type=Path,
        default=REPO_ROOT / "data" / "bts_v12_prepared",
    )
    parser.add_argument(
        "--validation_root",
        type=Path,
        default=REPO_ROOT / "data" / "bts_v12_validation",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "output" / "bts_v12_ablation",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=["control_v11", "fullres", "quality"],
    )
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    options = parser.parse_args()

    records = []
    for experiment in options.experiments:
        config = EXPERIMENTS[experiment]
        final_iteration = int(load_bts_geogs_config(config)["iterations"])
        for fold in options.folds:
            for seed in options.seeds:
                for scene in options.scenes:
                    source = _source_for(
                        scene, options.data_root, options.prepared_root).resolve()
                    split_path = (
                        options.validation_root
                        / f"fold_{fold}"
                        / scene
                        / "validation_split.json"
                    ).resolve()
                    if not source.is_dir():
                        raise FileNotFoundError(source)
                    if not split_path.is_file():
                        raise FileNotFoundError(
                            f"{split_path}; run tools/prepare_v12_validation.py first")
                    model_path = (
                        options.output_root
                        / experiment
                        / f"fold_{fold}"
                        / f"seed_{seed}"
                        / scene
                    ).resolve()
                    summary = _read_summary(model_path)
                    final_ply = (
                        model_path
                        / "point_cloud"
                        / f"iteration_{final_iteration}"
                        / "point_cloud.ply"
                    )
                    if summary is None or not final_ply.is_file():
                        checkpoint = _latest_checkpoint(model_path, final_iteration)
                        command = _training_command(
                            options.python,
                            config,
                            source,
                            model_path,
                            split_path,
                            final_iteration,
                            seed,
                            checkpoint,
                        )
                        if options.dry_run:
                            print(subprocess.list2cmdline(command))
                        else:
                            run_logged(command, model_path / "train.log")
                            summary = _read_summary(model_path)
                    records.append({
                        "experiment": experiment,
                        "fold": fold,
                        "seed": seed,
                        "scene": scene,
                        "model_path": str(model_path),
                        "summary": summary,
                    })

    result = _aggregate(records)
    options.output_root.mkdir(parents=True, exist_ok=True)
    result_path = options.output_root / "ablation_summary.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
