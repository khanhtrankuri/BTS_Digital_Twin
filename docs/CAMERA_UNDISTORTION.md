# Camera model and offline undistortion

## Goal

Prevent `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, and `FULL_OPENCV` coefficients from being silently discarded while keeping the CUDA rasterizer unchanged.

## Design

`CameraInfo` and runtime `Camera` preserve `camera_model`, `fx`, `fy`, `cx`, `cy`, and distortion. `validate_camera_intrinsics` checks model, dimensions, focal lengths, principal point, and finite coefficients. Raw distorted scenes rendered by the pinhole rasterizer emit a warning.

For SIMPLE_RADIAL, OpenCV receives `[k1, 0, 0, 0, 0]`; RADIAL receives `[k1, k2, 0, 0, 0]`. Images, feature observations, masks, depth, and normals use a shared mapping. Normals are renormalized and masks/depth use nearest interpolation.

## Prepare data

```powershell
python tools/prepare_undistorted_scene.py `
  --source C:\Users\Lenovo\Documents\Val_Race\HCM0421 `
  --output C:\data\Val_Race_undistorted\HCM0421 `
  --alpha 0.0 --crop_mode valid --copy_sparse `
  --process_depth --process_normal --process_masks
```

The source is never overwritten. Output cameras are `PINHOLE`; extrinsics and points3D are unchanged. `undistortion_metadata.json` records both matrices, coefficients, ROI, and dimensions.

## Visual verification

```powershell
python tools/visualize_undistortion.py `
  --source C:\Users\Lenovo\Documents\Val_Race\HCM0421 `
  --prepared C:\data\Val_Race_undistorted\HCM0421 `
  --output C:\data\undistortion_qa\HCM0421
```

Panels contain original, undistorted, displacement magnitude, valid pixels, and an undistorted grid.

## Return to raw output pixels

```powershell
python tools/redistort_render.py `
  --input output/HCM0421/test/ours_45000/renders `
  --output output/HCM0421/test_redistorted `
  --metadata C:\data\Val_Race_undistorted\HCM0421\undistortion_metadata.json
```

## Ablation

Compare A0 and A1 on HCM0421/HCM0674. Report outer-region edge F1, error by image radius, thin-wire recall, and global metrics. Do not mix different output grids.

## Limitations and rollback

The Phase-1 test CSV has no distortion coefficient; the training camera calibration is applied scene-wide. This assumes a shared camera. Use raw data with `CAMERA.USE_UNDISTORTED_DATA: false` to roll back; the warning remains unless explicitly disabled.

