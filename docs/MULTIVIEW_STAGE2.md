# Shared multiview Stage 2

## Status and gate

This phase is designed but intentionally not enabled. Phase 1/2 now provide full intrinsics, metric z-depth reprojection, alpha, normals, and uncertainty. Implementation should start only after strict Stage-1 validation checkpoints exist and paired source selection can be verified on real data.

## Proposed design

```text
target Stage-1 RGB/depth/normal/alpha/uncertainty
                       +
2–3 sharp, overlapping source RGB/features/depth
                       |
        shared lightweight encoder (1/2, 1/4)
                       |
     z-depth warp + visibility/depth confidence
                       |
       confidence-weighted feature aggregation
                       |
             shared NAFNet decoder
                       |
 bounded residual * correction gate + Stage-1 RGB
```

Source selection must combine center distance, orientation, frustum overlap, sharpness, and exposure mismatch. `utils/depth_reprojection.py` is the single coordinate implementation; Stage 2 must not duplicate it.

Output is constrained to:

```text
I_out = I_stage1 + gate * max_residual * tanh(delta)
```

The gate is zero for sky, out-of-image samples, occlusion, insufficient depth confidence, and high uncertainty. Identity fallback is mandatory.

## Losses

Charbonnier/L1, SSIM, LPIPS, supported-edge loss, view consistency, residual magnitude, and identity loss in unsupported regions. A new edge is supported only when a valid warped source edge agrees geometrically.

## Expected outputs and ablation

Compare A11 single-image with A12 shared multiview using the same strict split, parameter budget, and source data. Report hard bins, unsupported-new-edge rate, correction in sky/occlusion, VRAM, and inference time.

## Known limitations and rollback

The existing `stage2_refiner/multiview.py` is only a primitive and uses a separate convention (`align_corners=True`); it must be replaced by the shared tested projection utility before activation. Until A12 beats A11, use the single-image refiner or Stage 1 only.

