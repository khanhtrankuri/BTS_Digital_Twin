#!/usr/bin/env bash

# Production wrapper for a single RTX 4090 Linux server.
# The core dataset preparation, training, rendering, validation and packaging
# remain in run.sh; this file adds server guards, logging and CUDA tuning.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run_server.sh /path/to/Val_Race [extra pipeline arguments]

or:
  DATA_ROOT=/path/to/Val_Race ./run_server.sh [extra pipeline arguments]

Recommended:
  conda activate BTS
  tmux new -s bts4090
  ./run_server.sh /data/Val_Race

Server environment variables:
  GPU_ID              Physical NVIDIA GPU index (default: 0)
  MIN_VRAM_MIB        Required VRAM in MiB (default: 22000)
  ALLOW_NON_4090      Set to 1 to allow another >= MIN_VRAM_MIB GPU
  LOG_DIR             Log directory (default: output/server_logs)
  RUN_NAME            Log/lock name (default: bts_v11_gpu<GPU_ID>)
  MAX_JOBS            CUDA build parallelism (default: min(nproc, 12))
  OMP_NUM_THREADS     CPU threads used by PyTorch/OpenMP (default: min(nproc, 8))

All variables supported by run.sh are also supported, including DATA_ROOT,
PYTHON_BIN, PREPARED_ROOT, MODEL_ROOT, RENDER_ROOT, ZIP_PATH, GPU_PROFILE and
BUILD_EXTENSIONS.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$SCRIPT_DIR/run.sh" ]]; then
  echo "ERROR: Missing core runner: $SCRIPT_DIR/run.sh" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Install/activate the NVIDIA driver." >&2
  exit 2
fi

GPU_ID="${GPU_ID:-0}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-22000}"
ALLOW_NON_4090="${ALLOW_NON_4090:-0}"
GPU_PROFILE="${GPU_PROFILE:-rtx4090_24gb}"
BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-1}"
RUN_NAME="${RUN_NAME:-bts_v11_gpu${GPU_ID}}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/output/server_logs}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_ID must be a non-negative integer: $GPU_ID" >&2
  exit 2
fi
if [[ ! "$MIN_VRAM_MIB" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MIN_VRAM_MIB must be a non-negative integer: $MIN_VRAM_MIB" >&2
  exit 2
fi

gpu_row="$(
  nvidia-smi \
    --id="$GPU_ID" \
    --query-gpu=name,memory.total \
    --format=csv,noheader,nounits |
    head -n 1
)"
if [[ -z "$gpu_row" || "$gpu_row" != *,* ]]; then
  echo "ERROR: Cannot query NVIDIA GPU index $GPU_ID." >&2
  exit 2
fi

gpu_name="${gpu_row%,*}"
gpu_vram_mib="${gpu_row##*,}"
gpu_name="$(printf '%s' "$gpu_name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
gpu_vram_mib="$(printf '%s' "$gpu_vram_mib" | tr -d '[:space:]')"

if [[ ! "$gpu_vram_mib" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Invalid VRAM value returned by nvidia-smi: $gpu_vram_mib" >&2
  exit 2
fi
if (( gpu_vram_mib < MIN_VRAM_MIB )); then
  echo "ERROR: GPU has ${gpu_vram_mib} MiB; at least ${MIN_VRAM_MIB} MiB is required." >&2
  exit 2
fi
if [[ "$gpu_name" != *"4090"* && "$ALLOW_NON_4090" != "1" ]]; then
  echo "ERROR: Expected RTX 4090, found '$gpu_name'." >&2
  echo "Set ALLOW_NON_4090=1 only if this is intentional." >&2
  exit 2
fi

cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
if [[ ! "$cpu_count" =~ ^[0-9]+$ ]] || (( cpu_count < 1 )); then
  cpu_count=8
fi
default_max_jobs=$((cpu_count < 12 ? cpu_count : 12))
default_omp_threads=$((cpu_count < 8 ? cpu_count : 8))

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export GPU_PROFILE
export BUILD_EXTENSIONS
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export MAX_JOBS="${MAX_JOBS:-$default_max_jobs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$default_omp_threads}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

mkdir -p "$LOG_DIR"
timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
log_file="$LOG_DIR/${RUN_NAME}_${timestamp}.log"
lock_file="$LOG_DIR/${RUN_NAME}.lock"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "ERROR: Another '$RUN_NAME' process holds $lock_file." >&2
    exit 3
  fi
else
  echo "WARNING: flock is unavailable; duplicate-run protection is disabled." >&2
fi

ulimit -n 65535 2>/dev/null || true
exec > >(tee -a "$log_file") 2>&1

started_epoch="$(date +%s)"

finish() {
  status=$?
  finished_epoch="$(date +%s)"
  elapsed=$((finished_epoch - started_epoch))
  echo
  echo "Finished UTC : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "Exit status  : $status"
  echo "Elapsed sec  : $elapsed"
  echo "Log          : $log_file"
  nvidia-smi --id="$GPU_ID" \
    --query-gpu=name,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || true
}
trap finish EXIT

echo "BTS AbsGS server run"
echo "Started UTC  : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Host         : $(hostname)"
echo "GPU          : $gpu_name (${gpu_vram_mib} MiB)"
echo "CUDA device  : $CUDA_VISIBLE_DEVICES"
echo "GPU profile  : $GPU_PROFILE"
echo "MAX_JOBS     : $MAX_JOBS"
echo "OMP threads  : $OMP_NUM_THREADS"
echo "Allocator    : $PYTORCH_CUDA_ALLOC_CONF"
echo "Log          : $log_file"
echo

"$SCRIPT_DIR/run.sh" "$@"
