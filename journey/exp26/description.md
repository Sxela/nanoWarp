## exp26 — exp25 recipe, no LPIPS loss (flow only, ablation)

**Status: KILLED 2026-05-12 (step ~40k)**

Ablation: same as exp25 but `--lpips-weight 0.0`. Pure flow matching loss only.

Results at available checkpoints:

| step | lpips_sq | lpips_vgg | ssim  |
|------|----------|-----------|-------|
| 10k  | 0.174    | 0.316     | 0.607 |
| 20k  | 0.162    | 0.306     | 0.626 |
| 30k  | 0.172    | 0.309     | 0.625 |
| 40k  | 0.180    | 0.309     | 0.633 |

**Finding**: LPIPS loss is critical. Without it: −30% worse on lpips_vgg vs exp25
at every step, metrics plateau/regress after 20k (flow loss optimises reconstruction
but doesn't drive perceptual quality). VGG LPIPS is not redundant — it's the primary
driver of visual quality improvement. Run killed early; no need to see 80k.

---
