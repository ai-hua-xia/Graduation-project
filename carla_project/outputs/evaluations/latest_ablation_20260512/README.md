# Latest Ablation Summary

This directory keeps the compact final ablation result used by the current thesis manuscript.

It intentionally excludes bulky checkpoint snapshots, generated videos, smoke-test outputs,
supplemental I3D-FVD results, and left/right JS diagnostics. The main thesis table uses:

- `ar_psnr`
- `ar_ssim`
- `ar_lpips`
- `ar_fvd`
- `action_sens_mean_norm`
- `action_sensitivity_std_norm`

The same values are stored in `summary.csv` and `summary.json`.

The final-combination action-consistency values were recomputed on 2026-05-12
from the final checkpoint with seed 0 and 100 samples:

- `action_sens_mean_norm = 1.8242`
- `action_sensitivity_std_norm = 0.7598`

The autoregressive image/video metrics in that row remain from the retained
final-combination evaluation.
