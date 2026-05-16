### exp14 — exp10/exp12 architecture on the 1k-pair 1024px dataset — READY

The actual data-scaling test. Three different architectures at 128px all hit
LPIPS ≈ 0.152 — the ceiling is data-bound, not architecture-bound. exp14 is
the experiment that should break that ceiling.

Dataset (materialised 2026-05-10):

```
data/photo2anime_1k/
  train/source + train/target  (908 pairs, indices 000000..000907)
  val/source + val/target      (100 pairs, indices 000908..001007)
```

- 3.5× more pairs than the original (287 → 908 train).
- Source resolution is 1024px, much higher than the original — gives the
  augmentation pipeline real room for zoom-crop variations.
- Val split bumped from 50 → 100 → tighter confidence intervals on the
  metrics (val noise was ±2% with n=50; should drop to ±1.5% with n=100).

Augmentation knobs added (2026-05-10):
- `--aug-resize-scale` (default 1.10): intermediate resize ratio before
  random crop. With 1024px source images we can crank this much higher.
- `--aug-scale-jitter` (default 0.10): affine scale jitter around the crop.
  Higher value gives more zoom variation.

Suggested values for exp14 at 256px training:
- `--aug-resize-scale 1.5` → intermediate 384px (still well under 1024px
  source, plenty of crop variety).
- `--aug-scale-jitter 0.15` → effective scale range [0.85, 1.15] before crop.

Architecture: same as exp10 (no encoder, mc=88, FM, LPIPS aux 0.2, bf16,
dropout, full attention coverage). Resolution: 256px (per exp12 plan)
with `--attn-resolutions "16,32,64"` to maintain fractional attention
coverage.

Step count: 40k. More data gradient signal benefits from longer training;
the val-LPIPS curve from exp08-lpips/exp10 was still improving at 20k with
287 pairs, so with 3.5× more data the optimum step is likely deeper.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 40000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 1000 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp14_ds1k_256px_noenc_attn163264_bf16_mc88_40k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,1k-dataset,256px,exp14" `
    --outdir out/exp14_ds1k_256px_noenc_attn163264_bf16_mc88_40k
```

Wall-clock estimate: at 256px ~120-160 ms/step (per the exp12 estimate).
40k steps ≈ 80-100 min. With 1k-dataset's better gradient signal-to-noise
ratio, expect smoother convergence.

Predictions:
- LPIPS at 256px floor is 0.299 (vs 0.199 at 128px). exp14 LPIPS-vs-floor
  ratio should beat the 23% improvement we got at 128px-with-287-pairs.
  A relative improvement of 30-35% (LPIPS landing at 0.20-0.21) would
  confirm the data-scaling hypothesis.
- SSIM at 256px floor is 0.557; we'd expect exp14 SSIM around 0.72-0.76.
- **Visual shape complexity**: this is the headline question. The
  oversimplified-shapes failure mode should diminish with 3.5× more shape
  variants and 4× more pixels per shape.

If exp14 still has shape-simplicity issues, that's the trigger to
implement exp16 (GAN aux). Otherwise, exp14 is plausibly the
"best-current-recipe" run we'll show off.

---

## Known bugs / lessons

- **2026-05-11 lesson: vanilla GAN-from-scratch collapses to grid
  artifacts (exp20).** First exp20 used PatchGAN + hinge loss + spectral
  norm with `--gan-weight 0.1`, both G and D starting from random init
  and updating from step 1. Result: visually plausible 70-pixel patches
  that didn't tile coherently — periodic grid-like artifacts at the
  PatchGAN receptive-field scale. The training loop "succeeded" by loss
  numbers but the output was unusable.
  Root cause: random D produces meaningless gradient direction for G
  in the early steps. G wanders into a "fool the discriminator
  per-patch" attractor before learning the basic photo→anime mapping.
  Once stuck, alternating updates can't escape.
  Fix: fastai's three-phase NoGAN approach. Phase 1 trains G on
  perceptual loss alone (G learns the task). Phase 2 trains D alone on
  (real, current-G-output) pairs (D calibrates). Phase 3 starts
  alternating G+D with a sensible starting point.
  See exp21 spec; flags `--gan-pretrain-g-steps` and
  `--gan-pretrain-d-steps`.



- **2026-05-10 bug: LPIPS aux silently disabled when FeatureLoss flags
  weren't set.** Introduced during the FeatureLoss wiring. The
  `aux_lpips = raw_lpips` assignment got scoped into the feature-loss
  block, so `--lpips-weight 0.2` alone produced `aux_lpips=None` and
  `lpips_loss = 0.0` for the entire run. The reported training-log
  `lpips` column showed exactly 0, which was the clue.
  Effect: first exp14 launch trained as "FM only, no perceptual aux" —
  the slow-convergence regime characterised in exp07b.
  Fix: re-scoped the lpips wrapper construction inside `if args.lpips_weight > 0:`.
  **Lesson**: smoke tests must cover each flag *in isolation*, not just
  the "all flags on" path. Going forward, when adding any new aux loss,
  always test:
  1. Old flag only (e.g. `--lpips-weight 0.2`).
  2. New flag only (e.g. `--feature-content-weight 1.0 --feature-style-weight 5000`).
  3. Both together.

---
