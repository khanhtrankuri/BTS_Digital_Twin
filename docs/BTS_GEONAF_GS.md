# BTS GeoNAF-GS

`BTS GeoNAF-GS` is an optional two-stage architecture:

```text
RGB + camera
  -> frozen AbsGS/3DGS (geometry, visibility, SH, densification)
  -> Gaussian RGB + depth + normal + alpha + uncertainty + depth variance
  -> per-image robust geometry normalization
  -> one shared Geometry-Guided Residual NAFNet
  -> bounded refined RGB
```

Stage 1 remains the existing AbsGS v11 pipeline. `train.py`, `render.py`, old
configs and old checkpoints do not depend on Stage 2. `STAGE2.ENABLED` is
`false` by default.

## Refiner input and output

The default input order is:

```text
gaussian_rgb[3], normalized_depth[1], normal[3], alpha[1],
uncertainty[1], normalized_variance[1]
```

`DATA.INPUT_COMPONENTS` and `MODEL.IN_CHANNELS` are checked against one another.
Optional foreground, thin-structure, dynamic and sky/background masks can be
added later without changing the NAFNet implementation.

For valid pixels `alpha > ALPHA_THRESHOLD` and `depth > 0`:

```text
median = median(depth)
mad = median(abs(depth - median))
normalized_depth = clamp((depth - median) / (mad + eps), -clip, clip) / clip

relative_variance = depth_variance / (depth^2 + eps)
normalized_variance = clamp(log1p(relative_variance), 0, variance_clip)
                      / variance_clip
```

Images without enough valid depth return zero normalized maps. Statistics are
computed in float32; exported geometry is finite float16 NPZ.

The NAFNet produces four unconstrained channels:

```text
delta_rgb = tanh(output[0:3])
refine_mask = sigmoid(output[3:4])
effective_mask = refine_mask
```

With optional uncertainty gating:

```text
effective_mask = refine_mask * clamp(1 - uncertainty, MIN_MASK, 1)
```

The only final-image path is:

```text
final_rgb = clamp(
    gaussian_rgb + RESIDUAL_SCALE * effective_mask * delta_rgb,
    0, 1
)
```

The default `RESIDUAL_SCALE` is `0.15`. The final convolution is initialized to
zero, so a new checkpoint initially returns the Gaussian RGB exactly.

## Phase-2 objective

The implemented loss is:

```text
L_NAF =
    1.0   * Charbonnier(final, GT)
  + 0.2   * (1 - SSIM(final, GT))
  + 0.1   * L1(Sobel(final), Sobel(GT))
  + 0.01  * mean(abs(delta_rgb))
  + 0.005 * mean(refine_mask)
  + 0.05  * mean(exp(-k * RGB_error).detach() * abs(final - gaussian_rgb))
  + perceptual_weight * LPIPS(final, GT)
```

Every weight is configurable in `LOSS`. LPIPS defaults off. Training will not
download a perceptual backbone unless `--allow_weight_download` is explicitly
passed.

Optional multi-view loss uses camera-center/view-direction pairing, metric
z-depth forward reprojection, image-bound checks, target-depth occlusion,
alpha and uncertainty masks:

```text
L_mv = masked_L1(forward_warp(final_i, depth_i, camera_i, camera_j), final_j)
```

Multi-view is off by default. Because arbitrary random crops invalidate the
exported camera intrinsics, enabling it requires full-image, unflipped training
and batch size one.

## Export

One scene or a root of scenes is accepted:

```bash
python tools/export_stage2_dataset.py \
  --scene_root /kaggle/working/round2 \
  --model_root /kaggle/working/output/round2 \
  --output_root /kaggle/working/stage2_data \
  --iteration 15000 \
  --config configs/stage2/geonaf_base.yaml \
  --device cuda
```

The exporter only renders training cameras. It never exports private test RGB.
It resumes complete frames by default and writes:

```text
stage2_data/<scene>/
  rgb/*.png
  gt/*.png
  geometry/*.npz
  cameras/*.json
  manifest.json
```

Each NPZ contains float16 `depth`, `normal`, `alpha`, `uncertainty` and
`depth_variance`. The manifest contains deterministic train/validation splits,
dimensions, intrinsics, conventional world-to-camera extrinsics, camera center
and view direction.

## Train Phase 2

```bash
python train_stage2.py \
  --config configs/stage2/geonaf_base.yaml \
  --manifest_root /kaggle/working/stage2_data \
  --output_dir /kaggle/working/output_stage2/geonaf \
  --device cuda
```

One NAFNet is shared across every discovered scene. Phase 2 consumes detached
files, so no Gaussian parameter is present in the computation graph. Training
uses AdamW, cosine scheduling, AMP, gradient clipping, TensorBoard, resumable
checkpoints, `latest.pth`, `best_psnr.pth`, `best_ssim.pth` and optional early
stopping.

## Render

```bash
python render_with_refiner.py \
  -s /kaggle/working/round2/HCM0421 \
  -m /kaggle/working/output/round2/HCM0421 \
  --iteration 15000 \
  --refiner_checkpoint /kaggle/working/output_stage2/geonaf/best_psnr.pth \
  --refiner_config configs/stage2/geonaf_base.yaml \
  --output_dir /kaggle/working/final_render/HCM0421 \
  --device cuda
```

Outputs are separated into `raw`, `refined`, `mask`, `residual` and
`uncertainty`. Inference never reads GT. `--redistort_to_source_grid` uses the
same HCM redistortion metadata/function as the existing submission renderer.
Overlap-blended tiled inference defaults to `512` pixels with `64` pixels of
overlap.

## Evaluate

```bash
python evaluate_stage2.py \
  --manifest_root /kaggle/working/stage2_data \
  --refiner_checkpoint /kaggle/working/output_stage2/geonaf/best_psnr.pth \
  --config configs/stage2/geonaf_base.yaml \
  --output_dir /kaggle/working/evaluation_stage2 \
  --device cuda
```

The report contains per-image CSV, per-scene JSON, aggregate JSON and visual
comparisons. Raw and refined PSNR, SSIM, optional LPIPS, gradient error,
inference time and peak VRAM are reported separately. The aggregate explicitly
lists scene regressions and the fraction of pixels close to the maximum allowed
correction. No quality improvement should be claimed before these outputs have
been produced on real data.

## Optional Phase 3

Joint fine-tuning is guarded by `--enable_joint` and remains disabled in the
base config:

```bash
python joint_finetune.py \
  -s /data/HCM0421 \
  -m /models/HCM0421 \
  --iteration 15000 \
  --refiner_checkpoint /models/geonaf/best_psnr.pth \
  --config configs/stage2/geonaf_base.yaml \
  --output_dir /models/joint/HCM0421 \
  --enable_joint
```

It never calls densify, clone, split or prune; it asserts a constant Gaussian
count. Pose, blur and exposure are frozen. The direct Gaussian-to-GT objective
is always retained:

```text
L_total = L_GS_direct + lambda_final * L_NAF + lambda_mv * L_multiview
```

Position, SH and NAFNet gradient norms are logged separately.

## Tests and recommended ablations

Run:

```bash
python -m pytest -q \
  tests/test_geonaf_shapes.py \
  tests/test_geometry_normalization.py \
  tests/test_residual_refinement.py \
  tests/test_stage2_dataset.py \
  tests/test_stage2_multiview.py \
  tests/test_gaussian_frozen.py \
  tests/test_tiled_inference.py \
  tests/test_export_stage2.py \
  tests/test_stage2_training_smoke.py
```

Recommended controlled ablations:

1. RGB only (`INPUT_COMPONENTS: [gaussian_rgb]`, `IN_CHANNELS: 3`).
2. RGB + normalized depth.
3. RGB + all geometry.
4. Residual without/with the predicted mask.
5. Uncertainty gating off/on.
6. Edge loss off/on.
7. Multi-view loss off/on.
