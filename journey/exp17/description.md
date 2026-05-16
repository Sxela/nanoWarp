### exp17 — exp15 minus LPIPS-squeeze (VGG content + Gram only) — DONE (with spike)

Same as exp15 but `--lpips-weight 0.0`. Tests whether LPIPS-squeeze is
redundant with VGG-content L1.

**Hit a loss spike at step ~3k** (loss jumped from ~6 to ~21). Took ~5k
steps to recover. The model still finished 20k but effectively had ~13k
clean steps. This is what motivated the `--max-loss-spike-ratio`
safeguard added 2026-05-11.

Final val on the 1k val split:

| | lpips_squeeze | lpips_vgg | SSIM | notes |
|---|---:|---:|---:|---|
| exp14v2 (40k, LPIPS only) | 0.1832 | **0.2396** | **0.686** | LPIPS-squeeze in loop |
| exp15 (20k, LPIPS + VGG + Gram) | **0.1621** | 0.2926 | 0.631 | LPIPS-squeeze in loop |
| exp17 (20k, VGG + Gram only) | 0.1885 | 0.3369 | 0.600 | LPIPS-squeeze NOT in loop |

**Key findings (these reverse my earlier hypotheses):**

1. **LPIPS-squeeze was NOT redundant.** exp17 (dropping LPIPS, keeping
   VGG + Gram) is worst on *both* metrics. The BAPPS-learned weights on
   SqueezeNet features were carrying real gradient signal that VGG-content
   L1 doesn't replicate, despite VGG being the bigger backbone. So the
   "smaller backbone with learned weights" beats "bigger backbone without."

2. **exp14v2 (simplest recipe, LPIPS only) wins on lpips_vgg** — the
   out-of-loop metric. Adding VGG content + Gram to the training loss
   actively *hurts* out-of-loop perceptual quality. **Simpler training
   generalizes better here.**
   Caveat: exp14v2 trained 2× longer (40k vs 20k). Need to validate at
   step-20k checkpoint for fair comparison.

3. **SSIM tracks loss-complexity inversely.** Adding terms beyond
   LPIPS-squeeze hurts pixel-aligned structure. The Gram style loss in
   particular is position-unaware, so it pushes for "anime-textures-
   anywhere" rather than "anime-textures-aligned-with-target."

### Fair-step-count revalidation (added 2026-05-11)

exp14v2 validated at step-20k to remove the training-time advantage:

| at 20k | lpips_squeeze ↓ | lpips_vgg ↓ | SSIM ↑ |
|---|---:|---:|---:|
| **exp14v2 (LPIPS only)** | 0.1819 | **0.2481** | **0.674** |
| exp15 (LPIPS + VGG + Gram) | **0.1621** | 0.2926 | 0.631 |
| exp17 (VGG + Gram only) | 0.1885 | 0.3369 | 0.600 |
| exp14v2 (LPIPS only, 40k) | 0.1832 | 0.2396 | 0.686 |

**This reverses the exp15 "best" claim from earlier in this log.**

- exp14v2 wins lpips_vgg by **15%** and SSIM by **7%** at the same step count.
- exp15's lpips_squeeze advantage was a **metric-overfit artifact**:
  training with LPIPS-squeeze in the loss while reporting against
  LPIPS-squeeze produces an unrealistically good number.
- Out-of-loop perceptual quality (lpips_vgg) is best with the *simplest*
  training recipe.
- VGG content + Gram (whether alone in exp17 or alongside LPIPS in exp15)
  **actively hurt** quality on the honest metric.

### Revised "current best" recipe

**exp14v2** is the new best:
- FM (flow matching) + flow_sigma_noise 0.05
- no source encoder, mc=88
- 256px, attn 16/32/64, bf16
- LPIPS-squeeze aux at weight 0.2 (the only perceptual aux)
- source dropout 0.15
- aug resize_scale 1.5, scale_jitter 0.15
- AdamW lr 2e-4 → 1e-5 cosine, warmup 500
- grad clip 1.0, --max-loss-spike-ratio 10.0 (safeguard, future runs)
