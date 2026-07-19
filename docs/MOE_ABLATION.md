# Stage-2 lightweight MoE ablation

## Status

MoE is not implemented or enabled. The ablation runner rejects A13–A16 until A12 shared multiview has passed strict validation. This prevents an unverified capacity increase from being presented as progress.

## Proposed architecture

A shared encoder, geometry warp, fusion, and decoder surround three small patch-level experts at 1/4 resolution:

- Detail: active only for sharp source-supported, depth-consistent edges.
- Appearance: low-frequency exposure/color/shadow corrections; cannot create strong edges.
- Conservative: near-zero residual for extrapolation, sky, occlusion, or uncertainty.

Routing inputs are geometry confidence, uncertainty, edge support, visibility, sharpness, position/angle distance, and overlap. Scene ID is excluded. Start with soft routing and bounded expert-specific residuals.

## Required losses and logging

Unsupported-new-edge, identity in uncertain regions, router view consistency, light load balance, pseudo-route warmup, router TV, expert usage, entropy, activation by difficulty bin, route maps, and expert residual maps.

## Keep criterion

Keep MoE only if it improves strict validation by at least one requested threshold (0.5 dB global PSNR, 0.7–1.0 dB hard-bin PSNR, 0.015–0.02 LPIPS, or 5–8% relative thin F1) without degrading SSIM, depth/view consistency, unsupported edges, sky floaters, or hard-position quality. Improvement must occur on at least two hard scenes.

## Ablations and rollback

A13 is complete MoE; A14/A15/A16 remove Conservative/Detail/Appearance respectively. If criteria fail, remove MoE and retain A12. If A12 itself fails, retain A11 or Stage 1.

