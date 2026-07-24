#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh /path/to/Val_Race [extra Stage-1 pipeline arguments]

or:
  DATA_ROOT=/path/to/Val_Race ./run.sh [extra Stage-1 pipeline arguments]

Pipeline modes:
  PIPELINE_MODE=full     Train AbsGS, export Stage-2 data, train GeoNAF (default)
  PIPELINE_MODE=stage1   Run the original AbsGS v11 train/render/ZIP workflow
  PIPELINE_MODE=stage2   Reuse existing Gaussian checkpoints and train GeoNAF

Common environment variables:
  PYTHON_BIN         Python executable (default: python)
  BTS_SCENES         Space-separated scenes (default: all seven scenes)
  PREPARED_ROOT      Undistorted HCM data (default: data/bts_v11_prepared)
  MODEL_ROOT         Gaussian checkpoints (default: output/bts_v11_4090)
  GPU_PROFILE        auto, rtx4090_24gb, or memory_safe_8gb
  BUILD_EXTENSIONS   Build missing CUDA extensions: 1 or 0 (default: 1)

Stage-1 output variables:
  RENDER_ROOT        Rendered submission (default: submission_bts_v11)
  ZIP_PATH           Final ZIP (default: submission_bts_v11.zip)

Stage-2 variables:
  STAGE2_CONFIG          Config (default: configs/stage2/geonaf_base.yaml)
  STAGE2_DATA_ROOT       Export root (default: data/stage2_geonaf)
  STAGE2_OUTPUT_DIR      Shared refiner output (default: output/stage2_geonaf)
  STAGE2_ITERATION       Gaussian iteration to export (default: 15000)
  STAGE2_DEVICE          PyTorch device (default: cuda)
  STAGE2_RESOLUTION      Export resolution divisor (default: 1)
  STAGE2_RESUME          auto, none, or checkpoint path (default: auto)
  STAGE2_SKIP_EXPORT     Use existing manifests: 1 or 0 (default: 0)
  STAGE2_SKIP_TRAIN      Export only: 1 or 0 (default: 0)
  STAGE2_EXPORT_OVERWRITE  Re-export completed frames: 1 or 0 (default: 0)
  STAGE2_MAX_TRAIN_STEPS Optional smoke limit; 0 means full training
  STAGE2_ALLOW_WEIGHT_DOWNLOAD  Permit perceptual weights: 1 or 0 (default: 0)
  STAGE2_EXPOSURE_COMPENSATION Apply Stage-1 exposure during export (default: 0)

Examples:
  conda activate BTS

  # Full two-stage training:
  ./run.sh /data/Val_Race

  # Keep the historic AbsGS-only behavior:
  PIPELINE_MODE=stage1 ./run.sh /data/Val_Race

  # Resume only Stage 2 from latest.pth:
  PIPELINE_MODE=stage2 STAGE2_SKIP_EXPORT=1 ./run.sh /data/Val_Race

  # One-scene smoke run:
  BTS_SCENES="HCM0674" STAGE2_MAX_TRAIN_STEPS=10 ./run.sh /data/Val_Race
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
PIPELINE_MODE="${PIPELINE_MODE:-full}"
BTS_SCENES="${BTS_SCENES:-bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674}"
PREPARED_ROOT="${PREPARED_ROOT:-$SCRIPT_DIR/data/bts_v11_prepared}"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/output/bts_v11_4090}"
RENDER_ROOT="${RENDER_ROOT:-$SCRIPT_DIR/submission_bts_v11}"
ZIP_PATH="${ZIP_PATH:-$SCRIPT_DIR/submission_bts_v11.zip}"
GPU_PROFILE="${GPU_PROFILE:-rtx4090_24gb}"
BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-1}"

STAGE2_CONFIG="${STAGE2_CONFIG:-$SCRIPT_DIR/configs/stage2/geonaf_base.yaml}"
STAGE2_DATA_ROOT="${STAGE2_DATA_ROOT:-$SCRIPT_DIR/data/stage2_geonaf}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-$SCRIPT_DIR/output/stage2_geonaf}"
STAGE2_ITERATION="${STAGE2_ITERATION:-15000}"
STAGE2_DEVICE="${STAGE2_DEVICE:-cuda}"
STAGE2_RESOLUTION="${STAGE2_RESOLUTION:-1}"
STAGE2_RESUME="${STAGE2_RESUME:-auto}"
STAGE2_SKIP_EXPORT="${STAGE2_SKIP_EXPORT:-0}"
STAGE2_SKIP_TRAIN="${STAGE2_SKIP_TRAIN:-0}"
STAGE2_EXPORT_OVERWRITE="${STAGE2_EXPORT_OVERWRITE:-0}"
STAGE2_MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS:-0}"
STAGE2_ALLOW_WEIGHT_DOWNLOAD="${STAGE2_ALLOW_WEIGHT_DOWNLOAD:-0}"
STAGE2_EXPOSURE_COMPENSATION="${STAGE2_EXPOSURE_COMPENSATION:-0}"

case "$PIPELINE_MODE" in
  stage1|stage2|full) ;;
  *) die "PIPELINE_MODE must be stage1, stage2, or full: $PIPELINE_MODE" ;;
esac

require_binary_flag BUILD_EXTENSIONS "$BUILD_EXTENSIONS"
require_binary_flag STAGE2_SKIP_EXPORT "$STAGE2_SKIP_EXPORT"
require_binary_flag STAGE2_SKIP_TRAIN "$STAGE2_SKIP_TRAIN"
require_binary_flag STAGE2_EXPORT_OVERWRITE "$STAGE2_EXPORT_OVERWRITE"
require_binary_flag STAGE2_ALLOW_WEIGHT_DOWNLOAD "$STAGE2_ALLOW_WEIGHT_DOWNLOAD"
require_binary_flag STAGE2_EXPOSURE_COMPENSATION "$STAGE2_EXPOSURE_COMPENSATION"

[[ "$STAGE2_ITERATION" =~ ^[0-9]+$ ]] || {
  die "STAGE2_ITERATION must be a non-negative integer: $STAGE2_ITERATION"
}
[[ "$STAGE2_RESOLUTION" =~ ^[1-9][0-9]*$ ]] || {
  die "STAGE2_RESOLUTION must be a positive integer: $STAGE2_RESOLUTION"
}
[[ "$STAGE2_MAX_TRAIN_STEPS" =~ ^[0-9]+$ ]] || {
  die "STAGE2_MAX_TRAIN_STEPS must be a non-negative integer"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  die "Python executable not found: $PYTHON_BIN"
fi

DATA_ROOT="$(cd -- "$DATA_ROOT" 2>/dev/null && pwd)" || {
  die "Dataset root does not exist: $DATA_ROOT"
}

read -r -a SCENES <<<"$BTS_SCENES"
(( ${#SCENES[@]} > 0 )) || die "BTS_SCENES must contain at least one scene"

declare -A seen_scenes=()
for scene in "${SCENES[@]}"; do
  case "$scene" in
    bonsai|chair|HCM0421|HCM0539|HCM0540|HCM0644|HCM0674) ;;
    *) die "Unsupported scene in BTS_SCENES: $scene" ;;
  esac
  [[ -z "${seen_scenes[$scene]+present}" ]] || {
    die "Duplicate scene in BTS_SCENES: $scene"
  }
  seen_scenes["$scene"]=1
  [[ -d "$DATA_ROOT/$scene" ]] || die "Missing scene: $DATA_ROOT/$scene"
done

if [[ "$PIPELINE_MODE" == "stage2" && "$#" -gt 0 ]]; then
  die "Extra arguments are Stage-1-only and cannot be used with PIPELINE_MODE=stage2"
fi
if [[ "$PIPELINE_MODE" == "full" ]]; then
  for argument in "$@"; do
    [[ "$argument" != "--scenes" ]] || {
      die "Use BTS_SCENES instead of --scenes with PIPELINE_MODE=full"
    }
  done
fi
if [[ "$PIPELINE_MODE" != "stage1" && ! -f "$STAGE2_CONFIG" ]]; then
  die "Stage-2 config does not exist: $STAGE2_CONFIG"
fi

echo "Repository   : $SCRIPT_DIR"
echo "Dataset      : $DATA_ROOT"
echo "Python       : $PYTHON_BIN"
echo "Pipeline mode: $PIPELINE_MODE"
echo "Scenes       : ${SCENES[*]}"
echo "GPU profile  : $GPU_PROFILE"

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA.")
properties = torch.cuda.get_device_properties(0)
print(f"CUDA GPU     : {properties.name} ({properties.total_memory / 2**30:.1f} GB)")
PY

needs_rasterizer=0
if [[ "$PIPELINE_MODE" != "stage2" || "$STAGE2_SKIP_EXPORT" != "1" ]]; then
  needs_rasterizer=1
fi

if (( needs_rasterizer )) && ! "$PYTHON_BIN" -c \
  "import diff_gaussian_rasterization, simple_knn, fused_ssim" \
  >/dev/null 2>&1; then
  if [[ "$BUILD_EXTENSIONS" != "1" ]]; then
    die "CUDA extensions are missing and BUILD_EXTENSIONS=0"
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

needs_prepared_data=0
if [[ "$PIPELINE_MODE" != "stage2" || "$STAGE2_SKIP_EXPORT" != "1" ]]; then
  needs_prepared_data=1
fi

if (( needs_prepared_data )); then
  for scene in "${SCENES[@]}"; do
    [[ "$scene" == HCM* ]] || continue
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
fi

if [[ "$PIPELINE_MODE" == "stage1" || "$PIPELINE_MODE" == "full" ]]; then
  echo
  echo "=== Phase 1/3: AbsGS v11 train -> render -> validate -> ZIP ==="
  stage1_scene_args=()
  if [[ "$BTS_SCENES" != "bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674" ]]; then
    stage1_scene_args=(--scenes "${SCENES[@]}")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/tools/train_render_submission_v11.py" \
    --python "$PYTHON_BIN" \
    --data_root "$DATA_ROOT" \
    --prepared_root "$PREPARED_ROOT" \
    --model_root "$MODEL_ROOT" \
    --render_root "$RENDER_ROOT" \
    --zip_path "$ZIP_PATH" \
    --gpu_profile "$GPU_PROFILE" \
    --quiet \
    "${stage1_scene_args[@]}" \
    "$@"
fi

if [[ "$PIPELINE_MODE" == "stage2" || "$PIPELINE_MODE" == "full" ]]; then
  if [[ "$STAGE2_SKIP_EXPORT" != "1" ]]; then
    echo
    echo "=== Phase 2/3: Export frozen Gaussian renders and geometry ==="
    [[ -d "$STAGE2_DATA_ROOT" ]] || mkdir -p "$STAGE2_DATA_ROOT"
    for scene in "${SCENES[@]}"; do
      if [[ "$scene" == HCM* ]]; then
        scene_source="$PREPARED_ROOT/$scene"
      else
        scene_source="$DATA_ROOT/$scene"
      fi
      gaussian_checkpoint="$MODEL_ROOT/$scene/point_cloud/iteration_${STAGE2_ITERATION}/point_cloud.ply"
      [[ -f "$gaussian_checkpoint" ]] || {
        die "Missing Gaussian checkpoint for $scene: $gaussian_checkpoint"
      }
      export_args=(
        --scene_root "$scene_source"
        --model_root "$MODEL_ROOT"
        --output_root "$STAGE2_DATA_ROOT"
        --iteration "$STAGE2_ITERATION"
        --config "$STAGE2_CONFIG"
        --device "$STAGE2_DEVICE"
        --resolution "$STAGE2_RESOLUTION"
      )
      if [[ "$STAGE2_EXPORT_OVERWRITE" == "1" ]]; then
        export_args+=(--overwrite)
      fi
      if [[ "$STAGE2_EXPOSURE_COMPENSATION" == "1" ]]; then
        export_args+=(--exposure_compensation)
      fi
      "$PYTHON_BIN" "$SCRIPT_DIR/tools/export_stage2_dataset.py" \
        "${export_args[@]}"
    done
  else
    echo
    echo "Skipping Stage-2 export; using manifests under: $STAGE2_DATA_ROOT"
    for scene in "${SCENES[@]}"; do
      [[ -f "$STAGE2_DATA_ROOT/$scene/manifest.json" ]] || {
        die "Missing Stage-2 manifest: $STAGE2_DATA_ROOT/$scene/manifest.json"
      }
    done
  fi

  if [[ "$STAGE2_SKIP_TRAIN" != "1" ]]; then
    echo
    echo "=== Phase 3/3: Train one shared Geometry-Guided NAFNet ==="
    [[ -d "$STAGE2_OUTPUT_DIR" ]] || mkdir -p "$STAGE2_OUTPUT_DIR"
    train_args=(
      --config "$STAGE2_CONFIG"
      --manifest_root "$STAGE2_DATA_ROOT"
      --output_dir "$STAGE2_OUTPUT_DIR"
      --device "$STAGE2_DEVICE"
      --scenes "${SCENES[@]}"
    )
    case "$STAGE2_RESUME" in
      auto)
        if [[ -f "$STAGE2_OUTPUT_DIR/latest.pth" ]]; then
          echo "Resuming Stage 2: $STAGE2_OUTPUT_DIR/latest.pth"
          train_args+=(--resume "$STAGE2_OUTPUT_DIR/latest.pth")
        fi
        ;;
      none|"") ;;
      *)
        [[ -f "$STAGE2_RESUME" ]] || {
          die "STAGE2_RESUME checkpoint does not exist: $STAGE2_RESUME"
        }
        train_args+=(--resume "$STAGE2_RESUME")
        ;;
    esac
    if (( STAGE2_MAX_TRAIN_STEPS > 0 )); then
      train_args+=(--max_train_steps "$STAGE2_MAX_TRAIN_STEPS")
    fi
    if [[ "$STAGE2_ALLOW_WEIGHT_DOWNLOAD" == "1" ]]; then
      train_args+=(--allow_weight_download)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/train_stage2.py" "${train_args[@]}"
  else
    echo "Skipping Stage-2 training after export."
  fi
fi

echo
echo "Completed successfully."
if [[ "$PIPELINE_MODE" == "stage1" || "$PIPELINE_MODE" == "full" ]]; then
  if [[ "${#SCENES[@]}" -eq 7 ]]; then
    echo "Stage-1 ZIP : $ZIP_PATH"
    echo "Manifest    : ${ZIP_PATH%.*}.manifest.json"
  else
    echo "Stage-1 models: $MODEL_ROOT"
  fi
fi
if [[ "$PIPELINE_MODE" == "stage2" || "$PIPELINE_MODE" == "full" ]]; then
  echo "Stage-2 data : $STAGE2_DATA_ROOT"
  echo "Stage-2 model: $STAGE2_OUTPUT_DIR"
fi
