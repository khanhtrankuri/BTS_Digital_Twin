"""Train, render, verify, and package the metric-aligned BTS v12 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_submission import render_scene, zip_submission
from arguments import load_bts_geogs_config
from tools.train_render_submission_v11 import (
    EXPECTED_OUTPUTS,
    SCENES,
    detect_gpu_profile,
    run_logged,
    training_command,
    validate_output,
)
from utils.general_utils import safe_state


FINAL_ITERATION = 20_000


def newest_checkpoint(model_path: Path) -> Path | None:
    candidates = []
    for path in model_path.glob("chkpnt*.pth"):
        match = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
        if match and int(match.group(1)) < FINAL_ITERATION:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


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
        "--model_root",
        type=Path,
        default=REPO_ROOT / "output" / "bts_v12_fullres",
    )
    parser.add_argument(
        "--render_root",
        type=Path,
        default=REPO_ROOT / "submission_bts_v12",
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        default=REPO_ROOT / "submission_bts_v12.zip",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument(
        "--gpu_profile",
        choices=("auto", "memory_safe_8gb", "rtx4090_24gb"),
        default="auto",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Override automatic fullres.yaml / fullres_8gb.yaml selection.",
    )
    parser.add_argument(
        "--hcm_config",
        type=Path,
        help="Optional HCM-specific config; defaults to the 350k profile on 8 GB.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    safe_state(args.quiet)

    gpu_profile = detect_gpu_profile(args.gpu_profile)
    config = (
        args.config.resolve()
        if args.config
        else REPO_ROOT
        / "configs"
        / "bts_v12"
        / ("fullres.yaml" if gpu_profile == "rtx4090_24gb" else "fullres_8gb.yaml")
    )
    hcm_config = (
        args.hcm_config.resolve()
        if args.hcm_config
        else (
            REPO_ROOT / "configs" / "bts_v12" / "fullres_hcm_8gb.yaml"
            if gpu_profile == "memory_safe_8gb"
            else config
        )
    )
    for selected_config in {config, hcm_config}:
        if not selected_config.is_file():
            raise FileNotFoundError(selected_config)
        configured_iterations = int(
            load_bts_geogs_config(selected_config)["iterations"])
        if configured_iterations != FINAL_ITERATION:
            raise ValueError(
                f"v12 submission requires {FINAL_ITERATION} iterations, but "
                f"{selected_config} resolves to {configured_iterations}"
            )
    print(f"Selected GPU profile: {gpu_profile}", flush=True)
    print(f"Selected bonsai/chair config: {config}", flush=True)
    print(f"Selected HCM config: {hcm_config}", flush=True)

    started = time.perf_counter()
    models = {}
    for scene_name in args.scenes:
        scene_config = hcm_config if scene_name.startswith("HCM") else config
        source = (
            args.prepared_root / scene_name
            if scene_name.startswith("HCM")
            else args.data_root / scene_name
        ).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        model_path = (args.model_root / scene_name).resolve()
        final_ply = (
            model_path
            / "point_cloud"
            / f"iteration_{FINAL_ITERATION}"
            / "point_cloud.ply"
        )
        if not args.skip_train and not final_ply.is_file():
            checkpoint = newest_checkpoint(model_path)
            if gpu_profile == "memory_safe_8gb" and checkpoint is None:
                run_logged(
                    training_command(
                        args.python,
                        scene_config,
                        source,
                        model_path,
                        2000,
                        [2000],
                    ),
                    model_path / "auto_train.log",
                )
                checkpoint = model_path / "chkpnt2000.pth"
            run_logged(
                training_command(
                    args.python,
                    scene_config,
                    source,
                    model_path,
                    FINAL_ITERATION,
                    [7000, 12000, FINAL_ITERATION],
                    checkpoint,
                ),
                model_path / "auto_train.log",
            )
        if not final_ply.is_file():
            raise FileNotFoundError(final_ply)
        models[scene_name] = {
            "source": str(source),
            "model": str(model_path),
            "iteration": FINAL_ITERATION,
            "config": str(scene_config),
        }

    if not args.skip_render:
        for scene_name in args.scenes:
            spec = models[scene_name]
            render_scene(
                Path(spec["source"]),
                Path(spec["model"]),
                FINAL_ITERATION,
                1,
                args.render_root,
                exposure_compensation=False,
                test_exposure_mode="identity",
                redistort_to_source_grid=scene_name.startswith("HCM"),
            )

    if tuple(args.scenes) != SCENES:
        print("Partial scene run complete; ZIP requires all seven scenes.")
        return

    report = validate_output(args.render_root)
    zip_submission(args.render_root, args.zip_path)
    with zipfile.ZipFile(args.zip_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise IOError(f"ZIP CRC failure: {corrupt}")
        if len(archive.infolist()) != sum(value[0] for value in EXPECTED_OUTPUTS.values()):
            raise ValueError("ZIP does not contain the expected number of images")
    report.update({
        "candidate": "BTS v12 metric-aligned full-resolution refinement",
        "validation_required_before_submission": True,
        "baseline_leaderboard": {
            "score": 65.0192,
            "psnr": 22.665939,
            "ssim": 0.743145,
            "lpips": 0.271867,
        },
        "zip": str(args.zip_path.resolve()),
        "bytes": args.zip_path.stat().st_size,
        "sha256": hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_profile": gpu_profile,
        "models": models,
    })
    manifest = args.zip_path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
