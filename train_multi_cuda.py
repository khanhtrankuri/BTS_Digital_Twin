#
# Multi-GPU launcher for Gaussian Splatting.
#
# This file intentionally launches one train.py process per CUDA device instead
# of wrapping the Gaussian model with DDP. The upstream training loop mutates a
# single growing/pruning Gaussian set, so simple data parallel training of one
# scene is not a safe drop-in change. Process-per-GPU is the practical way to
# use Kaggle's 2x T4 setup for multiple scenes or multiple runs.
#

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def default_model_path(source_path, output_root):
    source = Path(source_path)
    name = source.name or source.parent.name or "scene"
    return str(Path(output_root) / name)


def is_scene_dir(path: Path):
    if (path / "train" / "sparse").exists() and (path / "test" / "test_poses.csv").exists():
        return True
    if (path / "sparse").exists():
        return True
    if (path / "transforms_train.json").exists():
        return True
    return False


def discover_scene_dirs(source_paths):
    discovered = []
    seen = set()

    for source_path in source_paths:
        path = Path(source_path).expanduser()
        if not path.exists():
            raise SystemExit(f"Source path does not exist: {source_path}")

        path = path.resolve()
        candidates = []
        if is_scene_dir(path):
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(
                child.resolve()
                for child in sorted(path.iterdir())
                if child.is_dir() and is_scene_dir(child)
            )

        if not candidates:
            raise SystemExit(f"No supported scenes found under: {path}")

        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key not in seen:
                discovered.append(candidate)
                seen.add(candidate_key)

    return discovered


def build_train_command(args, source_path, model_path, gpu_index, job_index):
    cmd = [
        args.python,
        args.train_script,
        "-s",
        source_path,
        "-m",
        model_path,
        "--port",
        str(args.port_base + job_index),
    ]

    if args.disable_viewer:
        cmd.append("--disable_viewer")

    cmd.extend(args.train_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return cmd, env


def validate_gpus(gpus, args):
    for gpu_index in gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        result = subprocess.run(
            [
                args.python,
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()
            message = (
                f"CUDA device '{gpu_index}' is not available to {args.python}. "
                "Use only visible device ids, for example '--gpus 0' on a single-GPU machine."
            )
            if detail:
                message += f"\nTorch error:\n{detail}"
            raise SystemExit(message)


def worker(worker_id, gpu_index, jobs, args, failures):
    while True:
        try:
            job_index, source_path, model_path = jobs.get_nowait()
        except queue.Empty:
            return

        cmd, env = build_train_command(args, source_path, model_path, gpu_index, job_index)
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[gpu {gpu_index}] job {job_index}: {printable}", flush=True)

        if args.dry_run:
            jobs.task_done()
            continue

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            failures.append((job_index, source_path, gpu_index, result.returncode))
            if args.stop_on_error:
                while True:
                    try:
                        jobs.get_nowait()
                        jobs.task_done()
                    except queue.Empty:
                        break

        jobs.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple Gaussian Splatting train.py runs across CUDA GPUs."
    )
    parser.add_argument(
        "-s",
        "--source_paths",
        nargs="+",
        required=True,
        help="One or more COLMAP/NeRF Synthetic dataset folders to train.",
    )
    parser.add_argument(
        "-m",
        "--model_paths",
        nargs="+",
        default=None,
        help="Optional output folders, one per source path.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated CUDA device ids. Kaggle 2xT4 usually uses 0,1.",
    )
    parser.add_argument(
        "--output_root",
        default="output/multigpu",
        help="Root used for auto-generated model paths when -m is omitted.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run train.py.",
    )
    parser.add_argument(
        "--train_script",
        default="train.py",
        help="Training script to launch.",
    )
    parser.add_argument(
        "--port_base",
        type=int,
        default=6010,
        help="Base viewer port. Each job uses port_base + job_index.",
    )
    parser.add_argument(
        "--enable_viewer",
        dest="disable_viewer",
        action="store_false",
        help="Enable the network viewer. Disabled by default for Kaggle/headless runs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the per-GPU commands without launching training.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Do not start more queued jobs after the first failure.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to train.py. Put them after --, for example: -- --iterations 7000 --eval",
    )
    parser.set_defaults(disable_viewer=True)
    args = parser.parse_args()

    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]

    args.source_paths = discover_scene_dirs(args.source_paths)

    print("Discovered scenes:", flush=True)
    for scene_path in args.source_paths:
        print(f"  {scene_path}", flush=True)

    gpus = parse_csv(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs were provided. Example: --gpus 0,1")

    if not args.dry_run:
        validate_gpus(gpus, args)

    if args.model_paths is not None and len(args.model_paths) != len(args.source_paths):
        raise SystemExit(
            "--model_paths must have the same length as discovered scenes. "
            f"Got {len(args.model_paths)} model paths for {len(args.source_paths)} scenes."
        )

    job_defs = []
    for job_index, source_path in enumerate(args.source_paths):
        source_path = str(source_path)
        model_path = (
            args.model_paths[job_index]
            if args.model_paths is not None
            else default_model_path(source_path, args.output_root)
        )
        job_defs.append((job_index, source_path, model_path))

    if args.dry_run:
        for job_index, source_path, model_path in job_defs:
            gpu_index = gpus[job_index % len(gpus)]
            cmd, _ = build_train_command(args, source_path, model_path, gpu_index, job_index)
            printable = " ".join(shlex.quote(part) for part in cmd)
            print(f"[gpu {gpu_index}] job {job_index}: {printable}", flush=True)
        print("\nDry run complete.", flush=True)
        return

    jobs = queue.Queue()
    for job_def in job_defs:
        jobs.put(job_def)

    failures = []
    threads = []
    for worker_id, gpu_index in enumerate(gpus):
        thread = threading.Thread(
            target=worker,
            args=(worker_id, gpu_index, jobs, args, failures),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if failures:
        print("\nFailed jobs:", flush=True)
        for job_index, source_path, gpu_index, returncode in failures:
            print(
                f"  job {job_index} on gpu {gpu_index}: {source_path} exited with {returncode}",
                flush=True,
            )
        raise SystemExit(1)

    print("\nAll jobs complete.", flush=True)


if __name__ == "__main__":
    main()
