# Validation and leakage protocol

## Goal

Evaluate pose generalization without test RGB and distinguish diagnostic full-sparse validation from strict train-only geometry.

## Create splits

```powershell
python tools/create_validation_split.py `
  --source C:\Users\Lenovo\Documents\Val_Race\HCM0421 `
  --output splits/HCM0421_temporal `
  --mode temporal_matched --ratio 0.10
```

Modes are `temporal_matched`, `position_extrapolation`, and `angular_extrapolation`. Each held-out image records its nearest train camera, normalized center distance, view angle, and difficulty bin.

```text
easy: position < 0.04 and angle < 8°
hard_position: position > 0.10
hard_angle: angle > 18°
extreme: both hard conditions
medium: all remaining samples
```

Analyze a split or the provided test-pose distribution with:

```powershell
python tools/analyze_pose_distribution.py `
  --source C:\Users\Lenovo\Documents\Val_Race\HCM0421 `
  --split splits/HCM0421_temporal/validation_split.json
```

## Strict train-only sparse model

The strict method needs the COLMAP feature database:

```powershell
python tools/rebuild_train_only_colmap.py `
  --source C:\Users\Lenovo\Documents\Val_Race\HCM0421 `
  --split splits/HCM0421_temporal/validation_split.json `
  --output strict_sparse/HCM0421 `
  --mode colmap --database C:\colmap\HCM0421\database.db
```

`filter_tracks` is only diagnostic: it removes holdout observations but cannot undo their previous influence on XYZ. The tool labels that output non-strict.

Train with original camera poses/images and strict points:

```yaml
VALIDATION:
  SPLIT_FILE: splits/HCM0421_temporal/validation_split.json
  STRICT_SPARSE_PATH: strict_sparse/HCM0421/sparse
```

## Reports

`leakage_report.json` contains holdout-observed points/tracks, point-count difference, and strict status. Metrics must be reported globally, per scene, per difficulty bin, and for center/outer/sky/edge/thin regions.

## Limitations and rollback

No test RGB is used. Test poses may characterize camera distribution only. Omit `VALIDATION.SPLIT_FILE` and `STRICT_SPARSE_PATH` to return to legacy LLFF/test handling.

