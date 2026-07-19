# BTS-GeoGS-v4 design and implementation status

## Goal

BTS-GeoGS-v4 improves novel-view generalization without test RGB or a generative prior. Every Stage-1 addition is opt-in at the argparse layer; the historical path remains active when flags are disabled.

## Implemented data flow

```text
COLMAP camera + image
  -> lossless intrinsic parsing and validation
  -> optional offline undistorted PINHOLE scene
  -> cached cameras at scheduled resolutions
  -> 3DGS render: RGB, z-depth, normal, alpha, uncertainty
  -> optional exposure / directional sky composition
  -> scheduled photometric and geometry losses
  -> persistent multi-view densification and protected pruning
```

The runtime coordinate convention is documented in `utils/camera_models.py` and `utils/depth_reprojection.py`. `R` is stored transposed for row-vector use. Depth is camera z-depth.

## Phase status

- Phase 1: implemented—camera metadata, offline undistortion/redistortion, exposure inference, pose-aware validation, and strict COLMAP tooling.
- Phase 2: implemented—progressive SH/resolution, persistent bitset support, burst suppression, optional depth consistency, masks, directional background, blur-aware sampling, Charbonnier loss, and thin protection.
- Phase 3: gated. The shared reprojection and uncertainty interfaces now exist, but the Stage-2 dataset does not yet supply verified paired source views and strict validation has not been trained on GPU.
- Phase 4: gated until Phase 3 beats the single-image baseline. A11–A16 intentionally fail in the runner instead of reporting an invalid comparison.

## Configuration

`configs/bts_v4/base.yaml` contains the full implementation. Scene configs inherit through `BASE`. Legacy configs still load without inheritance.

Scheduled image pyramids are cached on CPU by default (`RESOLUTION_SCHEDULE.CACHE_ON_CPU`) while transforms remain on GPU; only the sampled camera tensors are transferred each iteration. This avoids keeping three full copies of every scene image in VRAM.

Important independent switches are:

- `CAMERA.USE_UNDISTORTED_DATA`
- `SH_SCHEDULE.ENABLED`
- `RESOLUTION_SCHEDULE.ENABLED`
- `DENSIFICATION.METHOD`
- `DENSIFICATION.UNIQUE_VIEW_SUPPORT_ENABLED`
- `DENSIFICATION.BURST_SUPPRESSION_ENABLED`
- `PRUNING.THIN_STRUCTURE_PROTECTION`
- `SKY.ENABLED`
- `LOW_PARALLAX.ENABLED`
- `MULTIVIEW_DEPTH.ENABLED`

## Training

```powershell
python train.py -s <prepared-scene> -m output/HCM0421_v4 `
  --config configs/bts_v4/HCM0421.yaml --eval --disable_viewer
```

Expected outputs include the resolved argparse namespace, PLY checkpoints, `exposure.json`, optional `background.pt`, TensorBoard loss/schedule/densification histograms, and checkpoints with persistent state.

## Known limitations

- CUDA training quality has not been claimed or measured by this implementation pass.
- Offline undistorted test renders must be redistorted if the evaluator expects the raw pixel grid.
- Pairwise depth uses the closest camera center; an angle/overlap source selector belongs in Phase 3.
- Directional sky is low-order and cannot model transient clouds.
- Strict COLMAP retriangulation requires the original COLMAP feature database.

## Ablation and rollback

Use A0–A10 through `tools/run_ablation.py`. Set all v4 flags to false or use the existing `configs/bts_v3/full_hybrid.yaml` to return to the previous behavior. Old checkpoints remain loadable; missing persistent buffers initialize to zero.
