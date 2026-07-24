#!/usr/bin/env bash

# Production wrapper for a single RTX 4090 Linux server. run.sh owns the
# Stage-1/Stage-2 workflow; this file adds hardware guards, locking, logging,
# and CUDA allocator/thread tuning.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run_server.sh /path/to/Val_Race [extra Stage-1 pipeline arguments]

or:
  DATA_ROOT=/path/to/Val_Race ./run_server.sh [extra Stage-1 pipeline arguments]

Recommended full BTS GeoNAF-GS training:
  conda activate BTS
  tmux new -s bts-geonaf
  ./run_server.sh /data/Val_Race

Server environment variables:
  GPU_ID              Physical NVIDIA GPU index (default: 0)
  MIN_VRAM_MIB        Required VRAM in MiB (default: 22000)
  ALLOW_NON_4090      Set to 1 to allow another >= MIN_VRAM_MIB GPU
  LOG_DIR             Log directory (default: output/server_logs)
  RUN_NAME            Log/lock name (default: bts_geonaf_<mode>_gpu<GPU_ID>)
  MAX_JOBS            CUDA build parallelism (default: min(nproc, 12))
  OMP_NUM_THREADS     CPU threads used by PyTorch/OpenMP (default: min(nproc, 8))

All run.sh variables are supported. Important examples:
  PIPELINE_MODE       full (default), stage1, or stage2
  BTS_SCENES          Space-separated scene names
  MODEL_ROOT          Gaussian checkpoints
  STAGE2_DATA_ROOT    Exported Stage-2 dataset
  STAGE2_OUTPUT_DIR   Shared GeoNAF checkpoint directory
  STAGE2_RESUME       auto, none, or checkpoint path
  STAGE2_SKIP_EXPORT  Set 1 to train from existing manifests
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -f "$SCRIPT_DIR/run.sh" ]] || die "Missing core runner: $SCRIPT_DIR/run.sh"
command -v nvidia-smi >/dev/null 2>&1 || {
  die "nvidia-smi was not found. Install/activate the NVIDIA driver"
}

GPU_ID="${GPU_ID:-0}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-22000}"
ALLOW_NON_4090="${ALLOW_NON_4090:-0}"
GPU_PROFILE="${GPU_PROFILE:-rtx4090_24gb}"
BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-1}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
RUN_NAME="${RUN_NAME:-bts_geonaf_${PIPELINE_MODE}_gpu${GPU_ID}}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/output/server_logs}"

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || {
  die "GPU_ID must be a non-negative integer: $GPU_ID"
}
[[ "$MIN_VRAM_MIB" =~ ^[0-9]+$ ]] || {
  die "MIN_VRAM_MIB must be a non-negative integer: $MIN_VRAM_MIB"
}
[[ "$ALLOW_NON_4090" == "0" || "$ALLOW_NON_4090" == "1" ]] || {
  die "ALLOW_NON_4090 must be 0 or 1"
}
case "$PIPELINE_MODE" in
  stage1|stage2|full) ;;
  *) die "PIPELINE_MODE must be stage1, stage2, or full" ;;
esac

gpu_row="$(
  nvidia-smi \
    --id="$GPU_ID" \
    --query-gpu=name,memory.total \
    --format=csv,noheader,nounits |
    head -n 1
)"
if [[ -z "$gpu_row" || "$gpu_row" != *,* ]]; then
  die "Cannot query NVIDIA GPU index $GPU_ID"
fi

gpu_name="${gpu_row%,*}"
gpu_vram_mib="${gpu_row##*,}"
gpu_name="$(printf '%s' "$gpu_name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
gpu_vram_mib="$(printf '%s' "$gpu_vram_mib" | tr -d '[:space:]')"

[[ "$gpu_vram_mib" =~ ^[0-9]+$ ]] || {
  die "Invalid VRAM value returned by nvidia-smi: $gpu_vram_mib"
}
(( gpu_vram_mib >= MIN_VRAM_MIB )) || {
  die "GPU has ${gpu_vram_mib} MiB; at least ${MIN_VRAM_MIB} MiB is required"
}
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
export PIPELINE_MODE
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

echo "BTS GeoNAF-GS server run"
echo "Started UTC  : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Host         : $(hostname)"
echo "GPU          : $gpu_name (${gpu_vram_mib} MiB)"
echo "CUDA device  : $CUDA_VISIBLE_DEVICES"
echo "Pipeline mode: $PIPELINE_MODE"
echo "GPU profile  : $GPU_PROFILE"
echo "Stage-2 data : ${STAGE2_DATA_ROOT:-$SCRIPT_DIR/data/stage2_geonaf}"
echo "Stage-2 model: ${STAGE2_OUTPUT_DIR:-$SCRIPT_DIR/output/stage2_geonaf}"
echo "MAX_JOBS     : $MAX_JOBS"
echo "OMP threads  : $OMP_NUM_THREADS"
echo "Allocator    : $PYTORCH_CUDA_ALLOC_CONF"
echo "Log          : $log_file"
echo

"$SCRIPT_DIR/run.sh" "$@"
