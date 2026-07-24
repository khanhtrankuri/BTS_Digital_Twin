#!/usr/bin/env bash

# Auto-training entrypoint for:
#   - local RTX 4060 8 GB: sequential memory-safe Stage-1 scene jobs;
#   - Kaggle 2x T4 16 GB: two parallel Stage-1 scene queues.
# Both paths then export all selected scenes and train one shared GeoNAF model.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run_auto_train.sh [/path/to/Val_Race]

The dataset path can also be supplied through DATA_ROOT. With no path, local
mode tries ../Val_Race and Kaggle mode searches /kaggle/input.

Environment variables:
  TARGET_ENV          auto, local, or kaggle (default: auto)
  PYTHON_BIN          Python executable (auto-detects local Conda env BTS)
  BTS_SCENES          Space-separated scene list (default: all seven)
  WORK_ROOT           Writable output root
  MODEL_ROOT          Stage-1 Gaussian output
  PREPARED_ROOT       Writable undistorted HCM root
  STAGE2_DATA_ROOT    Exported Stage-2 data
  STAGE2_OUTPUT_DIR   Shared GeoNAF output
  STAGE2_CONFIG       Override automatic 8 GB/T4 config
  SKIP_STAGE1         Reuse existing Gaussian checkpoints: 1 or 0
  SKIP_STAGE2         Stop after Gaussian training: 1 or 0
  STAGE2_SKIP_EXPORT  Reuse existing Stage-2 manifests: 1 or 0
  STAGE2_SKIP_TRAIN   Export only: 1 or 0
  STAGE2_RESUME       auto, none, or checkpoint path
  STAGE2_MAX_TRAIN_STEPS  Optional smoke limit; 0 trains full epochs
  INSTALL_MISSING_DEPS    Install small Python dependencies if absent (default: 1)
  BUILD_EXTENSIONS    Build vendored CUDA extensions if absent (default: 1)

Local RTX 4060 examples from Git Bash:
  ./run_auto_train.sh
  ./run_auto_train.sh /d/Val_Race

Kaggle example:
  chmod +x run_auto_train.sh
  ./run_auto_train.sh /kaggle/input/<dataset>/Val_Race

Kaggle automatic dataset discovery:
  ./run_auto_train.sh

Smoke test one scene:
  BTS_SCENES="HCM0674" STAGE2_MAX_TRAIN_STEPS=10 \
    ./run_auto_train.sh /path/to/Val_Race
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_binary_flag() {
  local name="$1"
  local value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || {
    die "$name must be 0 or 1, got: $value"
  }
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET_ENV="${TARGET_ENV:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
BTS_SCENES="${BTS_SCENES:-bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
STAGE2_SKIP_EXPORT="${STAGE2_SKIP_EXPORT:-0}"
STAGE2_SKIP_TRAIN="${STAGE2_SKIP_TRAIN:-0}"
STAGE2_RESUME="${STAGE2_RESUME:-auto}"
STAGE2_MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS:-0}"
INSTALL_MISSING_DEPS="${INSTALL_MISSING_DEPS:-1}"
BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-1}"

for setting in \
  SKIP_STAGE1 SKIP_STAGE2 STAGE2_SKIP_EXPORT STAGE2_SKIP_TRAIN \
  INSTALL_MISSING_DEPS BUILD_EXTENSIONS; do
  require_binary_flag "$setting" "${!setting}"
done
[[ "$STAGE2_MAX_TRAIN_STEPS" =~ ^[0-9]+$ ]] || {
  die "STAGE2_MAX_TRAIN_STEPS must be a non-negative integer"
}
case "$TARGET_ENV" in
  auto|local|kaggle) ;;
  *) die "TARGET_ENV must be auto, local, or kaggle" ;;
esac

if [[ "$TARGET_ENV" == "auto" ]]; then
  if [[ -d /kaggle/input || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    TARGET_ENV="kaggle"
  else
    TARGET_ENV="local"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ "$TARGET_ENV" == "local" && -x "$HOME/anaconda3/envs/BTS/python.exe" ]]; then
    PYTHON_BIN="$HOME/anaconda3/envs/BTS/python.exe"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/python.exe" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/python.exe"
  else
    PYTHON_BIN="python"
  fi
fi

read -r -a SCENES <<<"$BTS_SCENES"
(( ${#SCENES[@]} > 0 )) || die "BTS_SCENES is empty"
declare -A seen_scenes=()
for scene in "${SCENES[@]}"; do
  case "$scene" in
    bonsai|chair|HCM0421|HCM0539|HCM0540|HCM0644|HCM0674) ;;
    *) die "Unsupported scene: $scene" ;;
  esac
  [[ -z "${seen_scenes[$scene]+present}" ]] || die "Duplicate scene: $scene"
  seen_scenes["$scene"]=1
done

discover_kaggle_data_root() {
  local probe_scene="${SCENES[0]}"
  local match=""
  local candidate=""
  [[ -d /kaggle/input ]] || return 1
  match="$(find /kaggle/input -maxdepth 5 -type d -name "$probe_scene" -print -quit)"
  [[ -n "$match" ]] || return 1
  candidate="$(dirname "$match")"
  for scene in "${SCENES[@]}"; do
    [[ -d "$candidate/$scene" ]] || return 1
  done
  printf '%s\n' "$candidate"
}

if [[ -z "${DATA_ROOT:-}" && -n "${1:-}" ]]; then
  DATA_ROOT="$1"
  shift
fi
(( $# == 0 )) || die "Unexpected arguments: $*"

if [[ -z "${DATA_ROOT:-}" && "$TARGET_ENV" == "local" ]] && \
   [[ -d "$SCRIPT_DIR/../Val_Race" ]]; then
  DATA_ROOT="$SCRIPT_DIR/../Val_Race"
fi
if [[ -z "${DATA_ROOT:-}" && "$TARGET_ENV" == "kaggle" ]]; then
  DATA_ROOT="$(discover_kaggle_data_root)" || {
    die "Could not discover the dataset under /kaggle/input; pass DATA_ROOT"
  }
fi
[[ -n "${DATA_ROOT:-}" ]] || {
  die "DATA_ROOT is required for local training"
}
DATA_ROOT="$(cd -- "$DATA_ROOT" 2>/dev/null && pwd)" || {
  die "Dataset root does not exist: $DATA_ROOT"
}
for scene in "${SCENES[@]}"; do
  [[ -d "$DATA_ROOT/$scene" ]] || die "Missing scene: $DATA_ROOT/$scene"
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  die "Python executable not found: $PYTHON_BIN"
}
command -v nvidia-smi >/dev/null 2>&1 || {
  die "nvidia-smi was not found"
}

mapfile -t GPU_NAMES < <(
  nvidia-smi --query-gpu=name --format=csv,noheader |
    sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
GPU_COUNT="${#GPU_NAMES[@]}"
(( GPU_COUNT > 0 )) || die "No NVIDIA GPU was detected"

if [[ "$TARGET_ENV" == "kaggle" ]]; then
  WORK_ROOT="${WORK_ROOT:-/kaggle/working/bts_geonaf}"
  DEFAULT_STAGE2_CONFIG="$SCRIPT_DIR/configs/stage2/geonaf_t4.yaml"
else
  WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR}"
  DEFAULT_STAGE2_CONFIG="$SCRIPT_DIR/configs/stage2/geonaf_8gb.yaml"
fi

MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/output/bts_v11_memory_safe}"
PREPARED_ROOT="${PREPARED_ROOT:-$WORK_ROOT/data/bts_v11_prepared}"
STAGE2_DATA_ROOT="${STAGE2_DATA_ROOT:-$WORK_ROOT/data/stage2_geonaf}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-$WORK_ROOT/output/stage2_geonaf}"
STAGE2_CONFIG="${STAGE2_CONFIG:-$DEFAULT_STAGE2_CONFIG}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/output/auto_train_logs}"

[[ -f "$STAGE2_CONFIG" ]] || die "Stage-2 config not found: $STAGE2_CONFIG"
for output_root in \
  "$MODEL_ROOT" "$PREPARED_ROOT" "$STAGE2_DATA_ROOT" \
  "$STAGE2_OUTPUT_DIR" "$LOG_ROOT"; do
  [[ -d "$output_root" ]] || mkdir -p "$output_root"
done

echo "BTS GeoNAF-GS automatic trainer"
echo "Environment   : $TARGET_ENV"
echo "Repository    : $SCRIPT_DIR"
echo "Dataset       : $DATA_ROOT"
echo "Python        : $PYTHON_BIN"
echo "GPUs          : $GPU_COUNT (${GPU_NAMES[*]})"
echo "Scenes        : ${SCENES[*]}"
echo "Work root     : $WORK_ROOT"
echo "Gaussian root : $MODEL_ROOT"
echo "Stage-2 data  : $STAGE2_DATA_ROOT"
echo "Stage-2 model : $STAGE2_OUTPUT_DIR"
echo "Stage-2 config: $STAGE2_CONFIG"

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA.")
print(
    "PyTorch CUDA  :",
    torch.__version__,
    torch.version.cuda,
    [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
)
PY

if ! "$PYTHON_BIN" -c \
  "import cv2, numpy, PIL, plyfile, tqdm, yaml" >/dev/null 2>&1; then
  [[ "$INSTALL_MISSING_DEPS" == "1" ]] || {
    die "Small Python dependencies are missing and INSTALL_MISSING_DEPS=0"
  }
  echo "Installing missing Python runtime dependencies..."
  "$PYTHON_BIN" -m pip install \
    opencv-python pillow plyfile pyyaml tqdm
fi

if ! "$PYTHON_BIN" -c \
  "import diff_gaussian_rasterization, simple_knn, fused_ssim" \
  >/dev/null 2>&1; then
  [[ "$BUILD_EXTENSIONS" == "1" ]] || {
    die "CUDA extensions are missing and BUILD_EXTENSIONS=0"
  }
  echo "Building vendored CUDA extensions once before worker launch..."
  if [[ -n "${CONDA_PREFIX:-}" && -z "${CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
  fi
  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps \
    "$SCRIPT_DIR/submodules/diff-gaussian-rasterization" \
    "$SCRIPT_DIR/submodules/simple-knn" \
    "$SCRIPT_DIR/submodules/fused-ssim"
fi

train_scene_queue() {
  local gpu_id="$1"
  shift
  local queue=("$@")
  local scene=""
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export PIPELINE_MODE=stage1
    export GPU_PROFILE=memory_safe_8gb
    export BUILD_EXTENSIONS=0
    export PYTHON_BIN
    export PREPARED_ROOT
    export MODEL_ROOT
    export RENDER_ROOT="$WORK_ROOT/submission_unused_gpu${gpu_id}"
    export ZIP_PATH="$WORK_ROOT/submission_unused_gpu${gpu_id}.zip"
    for scene in "${queue[@]}"; do
      echo
      echo "[GPU $gpu_id] Training Stage 1 scene: $scene"
      BTS_SCENES="$scene" "$SCRIPT_DIR/run.sh" "$DATA_ROOT" --skip_render
    done
  )
}

if [[ "$SKIP_STAGE1" != "1" ]]; then
  echo
  echo "=== Stage 1: memory-safe AbsGS scene training ==="
  if [[ "$TARGET_ENV" == "kaggle" && "$GPU_COUNT" -ge 2 ]]; then
    GPU0_SCENES=()
    GPU1_SCENES=()
    for index in "${!SCENES[@]}"; do
      if (( index % 2 == 0 )); then
        GPU0_SCENES+=("${SCENES[$index]}")
      else
        GPU1_SCENES+=("${SCENES[$index]}")
      fi
    done
    echo "GPU 0 queue: ${GPU0_SCENES[*]:-(empty)}"
    echo "GPU 1 queue: ${GPU1_SCENES[*]:-(empty)}"
    worker0_status=0
    worker1_status=0
    if (( ${#GPU0_SCENES[@]} > 0 )); then
      train_scene_queue 0 "${GPU0_SCENES[@]}" \
        >"$LOG_ROOT/stage1_gpu0.log" 2>&1 &
      worker0_pid=$!
    else
      worker0_pid=""
    fi
    if (( ${#GPU1_SCENES[@]} > 0 )); then
      train_scene_queue 1 "${GPU1_SCENES[@]}" \
        >"$LOG_ROOT/stage1_gpu1.log" 2>&1 &
      worker1_pid=$!
    else
      worker1_pid=""
    fi
    if [[ -n "$worker0_pid" ]]; then
      wait "$worker0_pid" || worker0_status=$?
    fi
    if [[ -n "$worker1_pid" ]]; then
      wait "$worker1_pid" || worker1_status=$?
    fi
    if (( worker0_status != 0 || worker1_status != 0 )); then
      echo "Stage-1 worker failure. Last GPU-0 log lines:" >&2
      tail -n 40 "$LOG_ROOT/stage1_gpu0.log" 2>/dev/null || true
      echo "Last GPU-1 log lines:" >&2
      tail -n 40 "$LOG_ROOT/stage1_gpu1.log" 2>/dev/null || true
      exit 1
    fi
    echo "Parallel Stage-1 queues completed."
    echo "Logs: $LOG_ROOT/stage1_gpu0.log and stage1_gpu1.log"
  else
    if [[ "$TARGET_ENV" == "kaggle" && "$GPU_COUNT" -lt 2 ]]; then
      echo "WARNING: Kaggle exposes only $GPU_COUNT GPU; using sequential mode." >&2
    fi
    for scene in "${SCENES[@]}"; do
      train_scene_queue 0 "$scene"
    done
  fi
else
  echo "Skipping Stage 1 and reusing checkpoints under: $MODEL_ROOT"
fi

for scene in "${SCENES[@]}"; do
  checkpoint="$MODEL_ROOT/$scene/point_cloud/iteration_15000/point_cloud.ply"
  [[ -f "$checkpoint" ]] || die "Missing completed Stage-1 checkpoint: $checkpoint"
done

if [[ "$SKIP_STAGE2" != "1" ]]; then
  echo
  echo "=== Stage 2: export geometry and train one shared GeoNAF ==="
  export CUDA_VISIBLE_DEVICES=0
  export PIPELINE_MODE=stage2
  export GPU_PROFILE=memory_safe_8gb
  export BUILD_EXTENSIONS=0
  export BTS_SCENES
  export PYTHON_BIN
  export PREPARED_ROOT
  export MODEL_ROOT
  export STAGE2_DATA_ROOT
  export STAGE2_OUTPUT_DIR
  export STAGE2_CONFIG
  export STAGE2_RESUME
  export STAGE2_SKIP_EXPORT
  export STAGE2_SKIP_TRAIN
  export STAGE2_MAX_TRAIN_STEPS
  "$SCRIPT_DIR/run.sh" "$DATA_ROOT"
else
  echo "Skipping Stage 2."
fi

echo
echo "Automatic training completed."
echo "Gaussian checkpoints: $MODEL_ROOT"
echo "Stage-2 manifests   : $STAGE2_DATA_ROOT"
echo "GeoNAF checkpoints  : $STAGE2_OUTPUT_DIR"
