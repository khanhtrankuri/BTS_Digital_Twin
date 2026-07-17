# Geometry-Guided Stage 2 Refiner

Stage 2 is standalone: it never changes Stage 1 Gaussian parameters or checkpoint files. The exporter freezes a loaded Stage 1 PLY, renders RGB plus metric camera-space depth, camera-space normal and alpha, and creates disjoint `train`/`val` manifest entries. Private test poses are never exported as training targets.

## PowerShell runbook

From the repository root, export every scene trained by `train_multi_cuda.py`:

```powershell
conda activate BTS
Set-Location 'C:\Users\Lenovo\Documents\gaussian-splatting'
python tools\export_stage2_dataset.py `
  -s 'C:\Users\Lenovo\Documents\phase1\private_set1' `
  -m 'output\multigpu_private' `
  --iteration 30000 `
  --output_dir 'data\stage2\private_set1' `
  --split 'train,val' `
  --val_ratio 0.10 `
  --split_seed 42 `
  --resolution 2
```

The exporter discovers every supported child scene, matches it to
`output\multigpu_private\<scene_name>`, writes one manifest under each scene,
and writes the combined manifest `data\stage2\private_set1\manifest.json`.
Exports run sequentially on the active CUDA device to avoid loading several
Stage 1 Gaussian models into the same GPU at once.

Train, evaluate the held-out Stage 2 views, and render target test poses:

```powershell
python train_stage2.py `
  --config configs\stage2\nafnet_base.yaml `
  --manifest data\stage2\private_set1\manifest.json `
  --output_dir output\stage2\private_set1

python evaluate_stage2.py `
  --config configs\stage2\nafnet_base.yaml `
  --checkpoint output\stage2\private_set1\best.pth `
  --manifest data\stage2\private_set1\manifest.json `
  --split val `
  --output_dir evaluation\private_set1

python render_with_refiner.py `
  -s 'C:\Users\Lenovo\Documents\phase1\private_set1\HNI0366' `
  -m 'output\multigpu_private\HNI0366' `
  --stage1_iteration 30000 `
  --refiner_checkpoint output\stage2\private_set1\best.pth `
  --output_dir output\final\HNI0366 `
  --split test `
  --resolution 1
```

Use `--disable_refiner` in the last command to produce the unchanged Stage 1 baseline. `rgb_only.yaml`, `rgb_depth.yaml`, `rgb_geometry.yaml`, and `rgb_geometry_edge.yaml` provide the single-view ablations.

## Validation policy

The default exporter hashes image names with `--split_seed` and holds out `--val_ratio` of GT cameras. It assigns globally unique camera IDs and validates that no `(scene, camera_id)` appears in two splits. `--use_official_val` uses Stage 1 test cameras only when their image files exist; private no-GT cameras are rejected as validation targets.

The optional multi-view warp primitives are implemented and identity/invalid/empty-mask tested, but paired-camera sampling is intentionally disabled in the shipped configs until camera/depth conventions are validated on a real scene. This keeps the verified single-view baseline unaffected.

## Tests

```powershell
python -m pytest tests -q
python tools\smoke_test_stage2.py --device cpu
```
