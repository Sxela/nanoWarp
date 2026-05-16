### exp08-noenc — drop source encoder, widen UNet, keep LPIPS — DONE

Following findings from exp07b + exp08-lpips:
- Encoder freezing fixed the optimizer collapse (exp05/06 → exp07b stable).
- LPIPS aux unlocked the actual stylization (exp07b → exp08-lpips).
- Open question: is the (frozen) ResNet18 encoder + multiscale fuse path
  contributing useful semantic priors, or is `cat([source, x_t])` at the input
  stem already enough?

exp08 tests this directly by dropping the encoder + fuses entirely and
widening the UNet to compensate (so total compute matches exp07b's ~42.7M).

After adding `--no-source-encoder` and refactoring `--model-ch` to scale the
whole UNet (widths are multiples of `model_ch`):

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --source-dropout 0.15 `
    --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --wandb --wandb-project nanoWarp --wandb-tags "flow,no-encoder,lpips,exp08" `
    --outdir out/exp08_noenc_lpips_mc88_20k
```

Param count: 43.9M total, all trainable (vs exp07b's 42.7M total with
11.2M frozen). UNet widths: 88 / 176 / 352 / 352 / 704.

Run launched 2026-05-10 morning, killed at step ~5000, resumed from
`model_step_005000.pt` after `--resume` shipped, killed again at step 17k
for GPU contention, resumed once more from `model_step_017000.pt` to finish.
Three wandb runs total: `exp08_noenc_lpips_mc88_20k`,
`exp08_noenc_lpips_mc88_20k_resumed`, `exp08_noenc_lpips_mc88_20k_resumed2`.

Final val (full 13 batches × bs=4 = 52 examples, EMA + 20-step Euler):

| metric | exp08-lpips (encoder on, mc=64) | **exp08-noenc (no encoder, mc=88)** | delta |
|---:|---:|---:|---:|
| mean_loss | — | 0.0053 | — |
| **SSIM ↑** | 0.719 | **0.734** | **+2.1%** |
| **LPIPS ↓** | 0.152 | **0.159** | **+4.3%** worse |
| total params | 42.7M | 43.9M | +3% (matched) |
| trainable params | 31.6M | **43.9M** | **+39%** (no-enc has more) |
| frozen params | 11.2M | 0 | — |

**The comparison is confounded by pretraining.**

The encoder's 11.2M frozen params are not "free architectural advantage" —
they encode ~1.28M ImageNet images × ~90 epochs of pretraining. That's
roughly **~115M training-example exposures** baked into those weights,
versus our paired-training budget of **20k × bs=4 = 80k examples**. Three
orders of magnitude more pretraining compute on the encoder side.

So the right reading of the result is **not** "encoder beat no-encoder at
fair size." It's "**11.2M ImageNet-pretrained params beat 11.2M extra
randomly-initialised params trained on 80k examples**." Given the data-scale
asymmetry, the surprise is that no-encoder closed *most* of the gap with
zero priors.

**Reframed conclusion:**
- The encoder's perceptual edge is mostly **amortised prior compute**, not
  an intrinsic architectural advantage of "having a separate encoder branch".
- At our current paired-data scale (287 pairs, 20k steps), pretraining priors
  cheaply buy ~3% LPIPS.
- For deployment / size-constrained scenarios: exp08-noenc is genuinely
  competitive — losing only ~3% LPIPS while being half the total size and
  having no external pretraining dependency.
- **Open question for exp10** (no-encoder + multi-scale attention): attention
  is a stronger inductive bias than conv, so it may close the prior-compute
  gap faster than a conv-only UNet can. Genuinely unclear what'll happen.

**Better-controlled future ablations** once the data-scale framing is clear:
- Train no-encoder for 100k+ steps to see if the gap closes with more
  paired-data exposure.
- Self-supervised pretraining of the no-encoder UNet's stem on a larger
  unpaired image set (e.g. ImageNet-100, COCO, LAION) before paired
  fine-tuning. Closer to a fair architectural comparison.
- Significantly larger paired dataset (1k-10k pairs) — the strongest
  intervention available.

**Train-val gap test (added 2026-05-10 evening):**

Ran validate.py twice on the exp08-noenc final EMA — once with `--split train`,
once with `--split val`. Same model, same sampler, same batch count (52 each).

| metric | train | val | gap |
|---|---:|---:|---:|
| mean_loss | 0.00494 | 0.00532 | val +7.7% worse |
| SSIM ↑ | 0.7424 | 0.7344 | val −1.1% worse |
| **LPIPS ↓** | **0.1443** | **0.1587** | **val +10% worse** |

Reading: the model **is not memorizing structure** (SSIM gap tiny), but **is
moderately overfitting perceptual features** (LPIPS gap 10%). Train SSIM is
only 0.74 — far from "fully fit the training set" — so we're also somewhat
capacity-limited on structural features. **Both more data and better
architecture would help, with data being the stronger lever for perceptual
quality.**

This is a useful diagnostic to repeat after each model change to see whether
new architecture options actually exploit the data we have or just reshape
the gap.

**Same test on exp08-lpips (encoder on, 2026-05-10 evening):**

| metric | exp08-noenc train | val | gap | exp08-lpips train | val | gap |
|---|---:|---:|---:|---:|---:|---:|
| mean_loss | 0.00494 | 0.00532 | +7.7% | 0.00488 | 0.00558 | +14% |
| SSIM ↑ | 0.7424 | 0.7344 | −1.1% | 0.7253 | 0.7194 | −0.8% |
| **LPIPS ↓** | **0.1443** | **0.1587** | **+10.0%** | **0.1366** | **0.1516** | **+11.0%** |

Key finding: **the encoder does not close the train-val gap.** Both models
have the same ~10% LPIPS generalization gap. The encoder shifts *both* train
and val LPIPS down by ~5% uniformly — it's a baseline-shift effect, not a
generalization effect.

**Reframe:**
- Encoder priors = ~5% LPIPS baseline shift across the board.
- 10% train-val LPIPS gap = data-scale tax that applies to both architectures
  equally.
- More paired data would help both models, not just no-encoder.
- SSIM is essentially saturated at this scale (~0.72-0.74 for both); the
  action is on perceptual quality.
- exp08-noenc actually has *slightly better SSIM* than exp08-lpips on both
  splits — the encoder pushes toward "ImageNet-feature-aligned texture" at
  small cost to pixel-level structural fidelity.

**For exp10**: attention's job is to recover the ~5% baseline shift without
external pretraining. Generalization gap is data-bound either way.

**Source-as-prediction floor (added 2026-05-10 evening):**

What you'd score on val/train if you literally output the source photo as
the predicted target. Any conditional model has to beat this.

| split | SSIM ↑ | LPIPS ↓ | image size |
|---|---:|---:|---:|
| train (287 pairs) | 0.6096 | 0.1976 | 128px |
| val (50 pairs) | 0.6168 | 0.1987 | 128px |
| train (287 pairs) | 0.5474 | 0.2972 | **256px** |
| val (50 pairs) | 0.5571 | 0.2993 | **256px** |

**Important calibration**: metric scales with image size. At 256px the
LPIPS floor is **~50% higher** than at 128px (0.299 vs 0.199), because
LPIPS gets more spatial bandwidth to penalise pair differences. Conversely
SSIM floor drops ~10%. Comparing absolute LPIPS values across resolutions
is misleading — compare *relative gap to floor*:

- 128px exp10 best: LPIPS 0.153 / floor 0.199 → **24% below floor** (gap 0.047)
- For an exp12 result to match in *relative* terms it'd need LPIPS ≈ 0.227
  at 256px (24% below 0.299). Anything substantially better than that is a
  real win at higher resolution; anything around 0.25-0.28 is roughly the
  same architectural ceiling we hit at 128px.

Reframed model performance as relative improvement over the floor:

| model | val SSIM | over floor | val LPIPS | over floor |
|---|---:|---:|---:|---:|
| floor | 0.617 | — | 0.199 | — |
| exp08-lpips | 0.719 | +17% | **0.152** | **−24%** |
| exp08-noenc | 0.734 | +19% | 0.159 | −20% |
| exp10 (step 20k) | **0.736** | **+19%** | 0.153 | −23% |

Useful frames this gives us:
- **The floors are nearly identical across splits** (0.198 vs 0.199 LPIPS),
  so our train-val gap isn't driven by val being structurally weirder.
- **Our absolute wins are modest** — best model is ~25% better than
  "do nothing" on LPIPS and ~20% better on SSIM. Plenty of headroom.
- **SSIM has less headroom than LPIPS** because source≈target structurally
  already gives SSIM ≈ 0.62. Future architecture improvements should be
  measured primarily on LPIPS.
- **The encoder's "+5% baseline shift"** = about a quarter of the total
  improvement-over-floor any model achieves. Not tiny.

### DataLoader perf (added 2026-05-10)

Default `--num-workers` bumped from 0 to 4. With `num_workers > 0` we
auto-enable `pin_memory` (CUDA only), `persistent_workers=True`, and
`prefetch_factor=2`. `.to(device, non_blocking=True)` on the data tensors.

Bench on the 4090 at bs=4 / 128px (`PairedImageDataset` with full augment):

| config | ms / batch | speedup |
|---|---:|---:|
| baseline (workers=0) | 83.0 | 1.0× |
| pin_memory + workers=0 | 272.8 | **0.3× — slower** |
| workers=2 | 139.5 | 0.6× |
| workers=4 (new default) | 53.6 | 1.5× |
| **workers=8** | **31.1** | **2.7×** |

Findings:
- **Footgun**: `pin_memory=True` with `num_workers=0` is *much slower* on
  Windows than no-pin baseline (3× slower). The trainer auto-disables
  pin_memory at workers=0.
- workers=8 brings batch loading down to ~step time (~30 ms). At that point
  GPU is the bottleneck again, which is what we want.
- Long runs should pass `--num-workers 8` explicitly (default 4 is the
  conservative cross-machine default).

### Weights & Biases (added 2026-05-10)

The trainer now logs to wandb when `--wandb` is passed. Captures:

- All CLI args as `wandb.config`.
- Git commit hash, short hash, branch, dirty flag (as `git_*` config keys).
- Param counts (total / trainable / frozen) and resolved UNet channels.
- Per-log-step: `train/loss`, `train/method_loss`, `train/lpips_loss`,
  `train/lr`, `train/grad_norm`, `train/t_low/high`.
- Per-panel-step: rendered panel as `train/panel` (wandb image).
- Per-val-step: `val/mean_loss_random_t`.
- Run summary: `final_loss`, `mean_loss_last_50`, `last_val_loss`.

Login once with `wandb login` then add `--wandb` to any train command.
Use `--wandb-mode offline` for offline runs (sync later with `wandb sync`).

Side-by-side comparison is the point — past runs (exp01–exp07b+LPIPS)
were not logged to wandb. From exp08 onward, runs will land in the
`nanoWarp` project for visual comparison.

---
