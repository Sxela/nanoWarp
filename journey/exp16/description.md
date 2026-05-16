### exp16 — VGG fastai per-layer feature weights `[5, 15, 2]`, no LPIPS — DONE

Run with VGG content + Gram using fastai's `[5, 15, 2]` per-layer weighting,
**no LPIPS aux** (so directly comparable to exp17, which used uniform layer
weights with the same other settings). 20k steps.

Final val on 1k val split:

| | lpips_sq ↓ | lpips_vgg ↓ | SSIM ↑ | loss config |
|---|---:|---:|---:|---|
| exp14v2 (20k) | 0.1819 | **0.2481** | **0.674** | LPIPS only |
| exp15 (20k) | **0.1621** | 0.2926 | 0.631 | LPIPS + VGG uniform |
| **exp16 (20k)** | **0.1671** | 0.3137 | 0.610 | **VGG fastai `[5,15,2]`, no LPIPS** |
| exp17 (20k) | 0.1885 | 0.3369 | 0.600 | VGG uniform, no LPIPS |
| exp14v2 (40k) | 0.1832 | 0.2396 | 0.686 | LPIPS only |

**Three findings:**

1. **Fastai weights `[5, 15, 2]` genuinely help vs uniform** in the VGG-only
   regime — exp16 beats exp17 by **11% on lpips_sq, 7% on lpips_vgg, 2% on
   SSIM**. Mid-layer emphasis (relu3_3 × 15) is real signal, not just
   folklore. If we ever return to the VGG path, use these weights.

2. **But the "VGG path is worse than LPIPS-only" conclusion stands.**
   exp14v2 still beats exp16 by **26% on lpips_vgg and 10% on SSIM**.
   Best-VGG-tuning loses to the simplest LPIPS-only recipe on the honest
   out-of-loop metric.

3. **lpips_sq disagrees with lpips_vgg on exp16 (a useful diagnostic).**
   exp16 (no LPIPS-squeeze in loss) beats exp14v2 (LPIPS-squeeze in loss)
   on lpips_sq by 8%. But exp14v2 wins lpips_vgg by 26%. When the two
   LPIPS metrics disagree, **trust lpips_vgg** — the bigger backbone and
   the out-of-loop status both favor it. exp16's lpips_sq win is likely a
   SqueezeNet-feature artifact (fastai weights happen to produce features
   that score well on SqueezeNet's specific BAPPS-tuned linear weights,
   independent of overall perceptual quality).

**Net**: the VGG-loss direction is cleanly crossed out as a defensible
conclusion (not just an inference from adjacent data). The two
high-priority next steps stay the same — push harder on the LPIPS path
(exp19, exp22) instead of variants of the losing direction.

After the exp17 revalidation showed VGG content + Gram underperformed
LPIPS-squeeze on the honest lpips_vgg metric, our prior was to skip
exp16. But "the metric-overfit story explains it" is one hypothesis;
"the layer weighting was wrong" is another. Running exp16 closes the
ambiguity:

- If exp16 *also* underperforms exp14v2 on lpips_vgg + SSIM, the VGG
  content + Gram path is genuinely worse for our task and we can cross
  it out cleanly.
- If exp16 surprisingly *beats* exp14v2 on lpips_vgg, mid-layer
  emphasis matters and the previous VGG runs (exp15, exp17) were
  poorly tuned.

Visual inspection is also informative — the LPIPS/SSIM numbers don't
capture whether stylization "looks right". A run with `[5, 15, 2]`
weights might produce qualitatively better outputs even if the
quantitative metrics don't move much.

Expected outcome (honest): lpips_vgg around 0.30-0.34 (somewhere
between exp15 and exp17), SSIM around 0.60-0.63. We expect to confirm
the VGG-path conclusion. But running it makes the conclusion *defensible*
rather than just *inferred*.

Single-variable change vs exp15. Adds per-VGG-layer weighting to the
content and style terms, matching fastai's `FeatureLoss` recipe. Default
in our impl was uniform `[1/n] * n`; this experiment tests whether
weighting middle layers (relu3_3) more heavily and the deepest layer
(relu4_3) less is a meaningful gain.

Layer weights `[5, 15, 2]` sum to 22 (vs uniform `[1/3, 1/3, 1/3]` summing
to 1.0), so the effective loss magnitude is ~22× higher. To keep the
overall content/style magnitude identical to exp15 — making this a clean
single-variable change about *layer weighting*, not loss magnitude — we
scale the overall content and style weights down by 22:

- `--feature-content-weight 0.045` (= 1.0 / 22, was 1.0 in exp15)
- `--feature-style-weight 227` (= 5000 / 22, was 5000 in exp15)
- `--feature-content-layer-weights "5,15,2"`
- `--feature-style-layer-weights "5,15,2"`

The relative emphasis is what changes: mid-level features (relu3_3) get
~68% of the total contribution (15/22) instead of 33% (uniform), at the
cost of low- and high-level features.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --feature-content-weight 0.045 --feature-style-weight 227 `
    --feature-loss-layers "8,15,22" `
    --feature-content-layer-weights "5,15,2" --feature-style-layer-weights "5,15,2" `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp16_fastai_layer_weights_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,feature-loss,gram-style,fastai-weights,bf16,1k-dataset,256px,exp16" `
    --outdir out/exp16_fastai_layer_weights_1k_256px_20k
```

Predictions:
- If exp16 LPIPS < exp15's 0.162 by > 2% → mid-layer emphasis is helpful
  for texture matching. Adopt fastai-style weighting as a new default.
- If exp16 ≈ exp15 → the per-layer weight choice doesn't move the
  needle for our task. Uniform is fine; one less hyperparam to think about.
- Visual check: deeper-layer de-emphasis (weight 2 vs 15 for the mid)
  *might* let more source semantics through, slightly improving identity
  preservation. Worth eyeballing alongside the metric.
