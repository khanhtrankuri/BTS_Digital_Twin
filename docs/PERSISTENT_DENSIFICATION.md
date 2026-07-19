# Persistent multi-view densification

## Goal

Select stable geometry corrections across windows instead of reacting to a single gradient, shadow edge, or blurred observation.

## State and formulas

Each Gaussian stores EMA score/hit state, recent-hit bitset, unique direction support bitset, depth/sky/low-parallax support, burstiness, last densification iteration, and persistent edge EMA. Twelve direction bins require one int64 per Gaussian.

Gradient burstiness is:

```text
B = max(E[g²] - E[|g|]², 0) / (E[g²] + eps)
```

Signals are robust IQR-normalized. The configurable score is:

```text
S = w_g S_grad + w_a S_abs + w_r S_residual + w_e S_edge
    + w_v S_multiview + w_d S_depth - w_b B
```

Sky and low-parallax multipliers are applied after scoring. A Gaussian must satisfy age, visibility, recent-hit, persistent-window, and optional unique-view/depth constraints.

## Configuration

See `DENSIFICATION` and `PRUNING` in `configs/bts_v4/base.yaml`. Key ablations are:

- A5: persistence.
- A6: persistence plus unique-view support.
- A7: add burst suppression.
- A9: add thin protected pruning.
- A10: complete Stage 1.

## Logging

TensorBoard records selected counts, signal intersections, clone/split/prune counts, score statistics, sky selections, low-parallax selections, Gaussian count, burstiness, unique views, persistent hits, and PSNR per million Gaussians.

## Expected behavior

Stable multi-view roof/antenna edges should accumulate support. One-view shadows, blur spikes, and sky edges should fail persistence or receive a multiplier. Thin protection requires anisotropy, edge EMA, view support, small projected area, and low burstiness.

## Known limitations

View bins measure directional diversity, not camera identity. Depth support is zero unless pairwise depth is enabled. True AbsGS still requires a rasterizer exposing `absgrad`; the code refuses to label an unavailable signal as true absolute gradient.

## Rollback

Set `DENSIFICATION.METHOD: hybrid`, disable unique-view/burst flags, and disable thin protection. The historical densification path is preserved.

