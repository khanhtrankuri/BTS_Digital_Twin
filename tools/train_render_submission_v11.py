"""Train, render, verify and package the BTS v11 AbsGS submission.

The selected recipe is deliberately simple and evidence-backed:
true absolute per-pixel gradient densification from iteration 500 to 2500,
then a frozen topology optimized until iteration 15000. It never reads test
RGB. Runs are sequential and resumable; RTX 4090 uses a continuous 15k run,
while the memory-safe profile restarts once at iteration 2k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_submission import render_scene, zip_submission
from utils.general_utils import safe_state


SCENES = ("bonsai", "chair", "HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")
FINAL_ITERATION = 15_000
EXPECTED_OUTPUTS = {
    "bonsai": (28, (1920, 1080)),
    "chair": (58, (720, 1280)),
    "HCM0421": (60, (1320, 989)),
    "HCM0539": (60, (1320, 989)),
    "HCM0540": (60, (1320, 989)),
    "HCM0644": (60, (1320, 989)),
    "HCM0674": (60, (1320, 989)),
}


def newest_checkpoint(model_path: Path) -> Path | None:
    candidates = []
    for path in model_path.glob("chkpnt*.pth"):
        match = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
        if match and int(match.group(1)) < FINAL_ITERATION:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def validate_output(output_dir: Path) -> dict:
    """Decode every expected JPEG and validate the competition layout."""

    report = {"total_images": 0, "scenes": {}}
    for scene_name, (expected_count, expected_size) in EXPECTED_OUTPUTS.items():
        scene_dir = output_dir / scene_name
        if not scene_dir.is_dir():
            raise FileNotFoundError(scene_dir)
        files = sorted(path for path in scene_dir.iterdir() if path.is_file())
        if len(files) != expected_count:
            raise ValueError(
                f"{scene_name}: expected {expected_count} images, found {len(files)}"
            )
        for path in files:
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(f"{scene_name}: non-JPEG output: {path.name}")
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != expected_size:
                    raise ValueError(
                        f"{path}: expected RGB {expected_size}, "
                        f"found {image.mode} {image.size}"
                    )
        report["scenes"][scene_name] = {
            "count": len(files),
            "size": list(expected_size),
        }
        report["total_images"] += len(files)
    if report["total_images"] != 386:
        raise ValueError(f"Expected 386 images, found {report['total_images']}")
    return report


def detect_gpu_profile(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        largest_mib = max(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
        return "rtx4090_24gb" if largest_mib >= 20_000 else "memory_safe_8gb"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "memory_safe_8gb"


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(subprocess.list2cmdline(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def training_command(
    python: str,
    config: Path,
    source: Path,
    model_path: Path,
    final_iteration: int,
    checkpoints: list[int],
    start_checkpoint: Path | None = None,
) -> list[str]:
    values = [str(value) for value in checkpoints]
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
        "--disable_viewer",
        "--test_iterations",
        "-1",
        "--save_iterations",
        *values,
        "--checkpoint_iterations",
        *values,
        "--psnr_max",
        "50",
        "--quiet",
    ]
    if start_checkpoint is not None:
        command.extend(["--start_checkpoint", str(start_checkpoint)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Lenovo\Documents\Val_Race"),
    )
    parser.add_argument(
        "--prepared_root", type=Path, default=REPO_ROOT / "data" / "bts_v11_prepared"
    )
    parser.add_argument(
        "--model_root", type=Path, default=REPO_ROOT / "output" / "bts_v11_absgrad"
    )
    parser.add_argument(
        "--render_root", type=Path, default=REPO_ROOT / "submission_bts_v11_absgrad"
    )
    parser.add_argument(
        "--zip_path", type=Path, default=REPO_ROOT / "submission_bts_v11_absgrad.zip"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument(
        "--gpu_profile",
        choices=("auto", "memory_safe_8gb", "rtx4090_24gb"),
        default="auto",
        help=(
            "4090 runs each scene in one process; the 8 GB profile restarts at "
            "iteration 2,000 to release CUDA allocator fragmentation."
        ),
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    safe_state(args.quiet)

    config = REPO_ROOT / "configs" / "bts_v11" / "absgrad_early_stop_15k.yaml"
    gpu_profile = detect_gpu_profile(args.gpu_profile)
    print(f"Selected GPU profile: {gpu_profile}", flush=True)
    started = time.perf_counter()
    models = {}
    for scene_name in args.scenes:
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
                # Dense AbsGS briefly approaches the 8 GB limit before the 2k
                # opacity prune. Restarting from this exact checkpoint releases
                # allocator fragmentation while preserving model and Adam state.
                run_logged(
                    training_command(
                        args.python, config, source, model_path, 2000, [2000]
                    ),
                    model_path / "auto_train.log",
                )
                checkpoint = model_path / "chkpnt2000.pth"
            run_logged(
                training_command(
                    args.python,
                    config,
                    source,
                    model_path,
                    FINAL_ITERATION,
                    [7000, FINAL_ITERATION],
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
        if len(archive.infolist()) != 386:
            raise ValueError(f"ZIP must contain 386 files, found {len(archive.infolist())}")
    report.update(
        {
            "zip": str(args.zip_path.resolve()),
            "bytes": args.zip_path.stat().st_size,
            "sha256": hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "models": models,
            "strict_validation": {
                "scene": "HCM0674 position holdout",
                "baseline_7k": 69.9796,
                "absgrad_7k": 72.40422964096069,
                "baseline_15k": 72.6666,
                "absgrad_15k": 73.11729192733765,
                "absgrad_30k": 73.11657667160034,
            },
            "gpu_profile": gpu_profile,
        }
    )
    manifest = args.zip_path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
