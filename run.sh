#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh /path/to/Val_Race [extra pipeline arguments]

or:
  DATA_ROOT=/path/to/Val_Race ./run.sh [extra pipeline arguments]

Environment variables:
  PYTHON_BIN         Python executable (default: python)
  PREPARED_ROOT      Undistorted HCM data (default: data/bts_v11_prepared)
  MODEL_ROOT         Checkpoints/models (default: output/bts_v11_4090)
  RENDER_ROOT        Rendered submission (default: submission_bts_v11)
  ZIP_PATH           Final ZIP (default: submission_bts_v11.zip)
  GPU_PROFILE        auto, rtx4090_24gb, or memory_safe_8gb
  BUILD_EXTENSIONS   Build missing CUDA extensions: 1 or 0 (default: 1)

Example:
  conda activate BTS
  ./run.sh /data/Val_Race
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${DATA_ROOT:-}" && -n "${1:-}" && "${1:0:1}" != "-" ]]; then
  DATA_ROOT="$1"
  shift
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "ERROR: DATA_ROOT is required." >&2
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_ROOT="${PREPARED_ROOT:-$SCRIPT_DIR/data/bts_v11_prepared}"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/output/bts_v11_4090}"
RENDER_ROOT="${RENDER_ROOT:-$SCRIPT_DIR/submission_bts_v11}"
ZIP_PATH="${ZIP_PATH:-$SCRIPT_DIR/submission_bts_v11.zip}"
GPU_PROFILE="${GPU_PROFILE:-rtx4090_24gb}"
BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

DATA_ROOT="$(cd -- "$DATA_ROOT" 2>/dev/null && pwd)" || {
  echo "ERROR: Dataset root does not exist: $DATA_ROOT" >&2
  exit 2
}

SCENES=(bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674)
HCM_SCENES=(HCM0421 HCM0539 HCM0540 HCM0644 HCM0674)

for scene in "${SCENES[@]}"; do
  if [[ ! -d "$DATA_ROOT/$scene" ]]; then
    echo "ERROR: Missing scene: $DATA_ROOT/$scene" >&2
    exit 2
  fi
done

echo "Repository : $SCRIPT_DIR"
echo "Dataset    : $DATA_ROOT"
echo "Python     : $PYTHON_BIN"
echo "GPU profile: $GPU_PROFILE"

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA.")
properties = torch.cuda.get_device_properties(0)
print(f"CUDA GPU   : {properties.name} ({properties.total_memory / 2**30:.1f} GB)")
PY

if ! "$PYTHON_BIN" -c \
  "import diff_gaussian_rasterization, simple_knn, fused_ssim" \
  >/dev/null 2>&1; then
  if [[ "$BUILD_EXTENSIONS" != "1" ]]; then
    echo "ERROR: CUDA extensions are missing and BUILD_EXTENSIONS=0." >&2
    exit 2
  fi
  echo "Building vendored CUDA extensions..."
  if [[ -n "${CONDA_PREFIX:-}" && -z "${CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
  fi
  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps \
    "$SCRIPT_DIR/submodules/diff-gaussian-rasterization" \
    "$SCRIPT_DIR/submodules/simple-knn" \
    "$SCRIPT_DIR/submodules/fused-ssim"
fi

for scene in "${HCM_SCENES[@]}"; do
  source_scene="$DATA_ROOT/$scene"
  prepared_scene="$PREPARED_ROOT/$scene"
  metadata="$prepared_scene/undistortion_metadata.json"

  if [[ -f "$metadata" ]]; then
    echo "Prepared HCM scene exists; skipping: $scene"
    continue
  fi
  if [[ -d "$prepared_scene" ]] && \
     [[ -n "$(find "$prepared_scene" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: Refusing to overwrite incomplete directory: $prepared_scene" >&2
    echo "Remove only that directory, then run this script again." >&2
    exit 2
  fi

  echo "Preparing undistorted HCM scene: $scene"
  "$PYTHON_BIN" "$SCRIPT_DIR/tools/prepare_undistorted_scene.py" \
    --source "$source_scene" \
    --output "$prepared_scene" \
    --copy_sparse
done

echo "Starting train -> render -> validate -> ZIP..."
"$PYTHON_BIN" "$SCRIPT_DIR/tools/train_render_submission_v11.py" \
  --python "$PYTHON_BIN" \
  --data_root "$DATA_ROOT" \
  --prepared_root "$PREPARED_ROOT" \
  --model_root "$MODEL_ROOT" \
  --render_root "$RENDER_ROOT" \
  --zip_path "$ZIP_PATH" \
  --gpu_profile "$GPU_PROFILE" \
  --quiet \
  "$@"

echo "Completed successfully."
echo "ZIP: $ZIP_PATH"
echo "Manifest: ${ZIP_PATH%.*}.manifest.json"
