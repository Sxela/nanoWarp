## exp53 — LANCZOS resize on exp50 recipe

**Status: DONE 2026-05-18** (negative result, exp52 stays canonical)

One-flag delta vs exp50: PIL resize filter for the source-pool
downscale switched from BILINEAR to LANCZOS on the "real" resize
paths (initial scaled zoom, val direct resize, post-crop fallback).
Affine (rotate/perspective) and corruption-aug paths kept BILINEAR.
Same architecture, same data, same recipe, 20k @ 256px bs=4.

Hypothesis: sharper input → finer prediction → better face metrics
on portraits.

**Results** (vs exp50 BILINEAR, same recipe):

| split | metric | exp50 | exp53 | Δ |
|---|---|---|---|---|
| val (legacy) | lpips_sq | 0.150 | **0.148** | -1% (tie) |
| val (legacy) | lpips_vgg | 0.297 | 0.303 | +2% (regress) |
| val (legacy) | ssim | 0.516 | 0.485 | **-6% (regress)** |
| val (legacy) | face_lpips_sq | 0.201 | 0.214 | **+6.5% (regress)** |
| val (legacy) | face_lpips_vgg | 0.379 | 0.402 | **+6% (regress)** |
| val (legacy) | face_ssim | 0.605 | 0.533 | **-12% (regress)** |
| val_portraits | lpips_sq | 0.170 | **0.164** | -3.5% (small win) |
| val_portraits | lpips_vgg | 0.353 | 0.355 | tie |
| val_portraits | ssim | 0.444 | 0.423 | -5% (regress) |
| val_portraits | **face_lpips_sq** | **0.124** | **0.124** | **0% (exact tie)** |
| val_portraits | face_lpips_vgg | 0.285 | 0.289 | +1% (tie) |
| val_portraits | face_ssim | 0.544 | 0.521 | -4% (regress) |
| val_portraits | Δ lpips_vgg | 0.037 | 0.039 | tie |

**Interpretation**: not the lever. Three signals point the same way:

1. **face_lpips_sq on val_portraits is identical** (0.124 → 0.124).
   The model wasn't bottlenecked on input sharpness — LPIPS-squeeze
   on FFHQ portraits at 512→256 doesn't discriminate between
   BILINEAR and LANCZOS source.

2. **SSIM regresses across the board** (-4% to -12%). LANCZOS
   overshoot near sharp edges produces small pixel-space variations
   that SSIM (luminance + structure) penalizes hard, even though
   the images look visually sharper. Known property of LANCZOS.

3. **Legacy val face metrics regress sharply** (face_lpips_sq
   +6.5%, face_ssim -12%). Legacy val has tiny / peripheral /
   non-frontal faces — at small pixel sizes, LANCZOS ringing
   amplifies edge noise rather than recovering detail. The tiny
   `lpips_sq -3.5%` win on portraits is overwhelmed by these
   regressions everywhere else.

**Conclusion**: BILINEAR was already adequate at 512→256. Sharpness
isn't the bottleneck; data diversity / training duration are. Do
**not** promote to 80k — exp52 stays the canonical baseline.

**What this rules out**: any "free win from better resize" hypothesis
at the current 256px target. If we move to 384/512 target later, the
downscale ratio shrinks and LANCZOS matters even less — so this
result implicitly closes that door too.

Script: `scripts/run_exp53_lanczos_at_exp50_recipe.sh`
Outdir: `out/exp53_lanczos_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k`

---
