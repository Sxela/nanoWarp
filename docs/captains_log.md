# Captain's log — img2img photo→anime

A chronological record of experiments run, with the exact command used for each.
Findings and reasoning live in [findings_2026-05-09.md](findings_2026-05-09.md);
this file is the lab notebook.

Dataset preparation was a one-time step:

```powershell
python scripts/prepare_photo2anime.py
# 337 paired files in anime_ds → data/photo2anime/{train,val}/{source,target}
# 287 train, 50 val, deterministic tail-50-by-index split
```

All commands below assume `.venv` is active and the environment is set with
`$env:PYTHONPATH = ".\nanoWarp"`.

---

## 2026-05-09

### exp01 — baseline (eps diffusion, partial freeze, 2k steps)

Reproduces docs/first_experiments.md step 1.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --outdir out/exp01_baseline
```

- 2000 steps, bs=4, 128px, lr=2e-4, EMA 0.999, ε prediction.
- Final loss ~0.057, val mean_loss ~0.012.
- Visually clean train-time x0_hat panels — but full DDIM-50 inference output was
  washed grey scribbles. Triggered the realization that train-time x0_hat at
  random t is not the same as full reverse sampling.

Validation (later replaced by the rewritten validate.py):

```powershell
python scripts/sample.py img2img-v1-val data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema `
    --outdir out/exp01_baseline_val
```

Sampler/clamp investigation runs on the same exp01 EMA checkpoint:

```powershell
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 5  `
    --outdir out/exp01_infer05
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 20 `
    --outdir out/exp01_infer20
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 50 `
    --outdir out/exp01_infer50
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 999 `
    --outdir out/exp01_infer999_ddim_clamp
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 1000 `
    --outdir out/exp01_infer1000_ddpm
```

All flavors (DDIM 5/20/50, DDIM 999 with clamp, full DDPM 1000) failed in
different ways. Concluded the issue was the **model**, not the sampler.

After rewriting validate.py to do full reverse sampling + high-t diagnostic:

```powershell
python scripts/sample.py img2img-v1-val data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 50 `
    --max-batches 4 --panel-count 2 --save-progress-strip `
    --outdir out/exp01_baseline_val_v2
```

Result: `mean_loss=0.0095, mean_ssim_sampled=0.353, mean_lpips_sampled=0.533`.
The high-t diagnostic column showed pure colored static — confirmed source
conditioning collapses at high t.

### exp02 — source-in-stem (2k steps)

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml --source-in-stem `
    --outdir out/exp02_source_in_stem
```

- Loss 0.026 (better than exp01 0.031), but SSIM 0.230 (worse) and LPIPS 0.689 (worse).
- Source-in-stem helps the random-t reconstruction loss but hurts structural
  similarity at this scale.

### exp03 — LPIPS aux loss 0.1 (2k steps)

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml --lpips-weight 0.1 `
    --outdir out/exp03_lpips_01
```

- LPIPS aux dropped from 0.87 → ~0.35 over the run. Did not validate further;
  pivoted to longer baselines after this.

### exp05 — long baseline (20k steps, no safety nets) — KILLED

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 `
    --outdir out/exp05_long_baseline_20k
```

- Diverged sharply at ~step 5000–6000. Loss floor jumped from ~0.02 to ~0.15
  and stayed elevated. Visible in panels: x0_hat clean through step 4k, grainy
  from step 5k+. Killed at step ~7000.

### exp06 — same run + every safety net we had (20k steps) — KILLED

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --prediction-type v --source-dropout 0.15 `
    --high-t-warmup-steps 2000 --high-t-warmup-low 500 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --outdir out/exp06_vpred_dropout_warmup_20k
```

- Stack: v-prediction + source dropout 0.15 + high-t warmup 2k + grad clip 1.0
  + LR warmup 500 + cosine decay to 1e-5.
- **Collapsed at the same step ~5000–6000** as exp05. Optimizer hygiene did
  not save it. Killed.
- This was the data point that pinned the cause on the trainable ResNet
  encoder layers (layer2/3/4 drift), not the optimizer.

### exp07 — flow matching + freeze=all + safety nets (20k steps) — KILLED EARLY

After adding `--method flow`, `--freeze-source-encoder all`, and the `eval()`
override on frozen stages:

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --freeze-source-encoder all --source-dropout 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --outdir out/exp07_flow_freeze_all_20k
```

- **Cleared step 5000 with no collapse.** Loss curve is a tight clean decay,
  no vertical jump. Confirmed the freeze hypothesis.
- Visually at step 5000, predicted-target panels showed recognizable anime
  stylization — qualitatively far above any diffusion run.
- Killed at step ~6000 only because we wanted to restart with intermediate
  checkpoint saving (the running process didn't have it).

### exp07b — same as exp07 but with mid-run checkpoints + full-sample panels (20k steps) — DONE

After adding `--checkpoint-every` and replacing the random-t panel with a
full-sample panel:

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --freeze-source-encoder all --source-dropout 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --outdir out/exp07b_flow_chkpt_20k
```

Mid-run validations (run in parallel as checkpoints land):

```powershell
python scripts/sample.py img2img-v1-val data/photo2anime `
    --checkpoint out/exp07b_flow_chkpt_20k/model_step_001000.pt --use-ema `
    --sample-steps 20 --max-batches 4 --panel-count 2 `
    --outdir out/exp07b_val_step_001000

# step-5000 watcher armed; will fire automatically when that checkpoint lands.
```

Val curve across the run:

| step | loss | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|
|  1k | 0.0177 | 0.616 | 0.219 |
|  5k | 0.0100 | 0.629 | 0.176 |
| 10k | 0.0029 | 0.648 | 0.190 |
| 15k | 0.0026 | 0.691 | 0.166 |
| 20k | 0.0025 | 0.701 | 0.159 |

- MSE loss had nearly plateaued by 15k, but **SSIM and LPIPS were still
  improving** at 20k. Pixel-MSE convergence is independent of perceptual
  quality at this scale — the model was still refining detail that doesn't
  show up in MSE.
- Visually, the predicted-target column was still mostly
  "source + mild palette shift" through step 4k; meaningful anime stylization
  was emerging slowly. Real flat-color stylization didn't really kick in
  until LPIPS aux was added (see exp07b+LPIPS below).

### exp08-lpips — exp07b config + LPIPS 0.2 (encoder still on) — DONE

Same FM + freeze=all + dropout + safety nets, plus `--lpips-weight 0.2`.
Hypothesis (from exp07b's "MSE plateaued / perceptual still climbing"
pattern): LPIPS aux can drive what MSE no longer can.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --freeze-source-encoder all --source-dropout 0.15 `
    --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --outdir out/exp07b_lpips_20k
```

- **Step-2k panel already shows real anime stylization** — flattened hair
  forms, anime-style eye treatment, smoothed skin / sharpened edges, palette
  shift. Identity / pose / composition preserved exactly.
- For comparison, exp07b (no LPIPS) at step 2k was barely stylized. LPIPS aux
  was the missing ingredient.
- Note for FM specifically: LPIPS aux lives on `x_target_hat = x_t + (1-t)·v_hat`,
  which is meaningful at every `t`. Unlike diffusion (where random-t
  `x0_hat` reconstruction is partly gameable through `x_t` leakage), FM's
  LPIPS target is non-gameable, so the aux signal converts directly to
  perceptual quality.
Final val curve (vs exp07b at every checkpoint):

| step | exp07b SSIM | **exp08-lpips SSIM** | exp07b LPIPS | **exp08-lpips LPIPS** |
|---:|---:|---:|---:|---:|
|  1k | 0.616 | **0.608** | 0.219 | **0.161** |
|  5k | 0.629 | **0.678** | 0.176 | **0.148** |
| 10k | 0.648 | **0.703** | 0.190 | **0.148** |
| 15k | 0.691 | **0.715** | 0.166 | **0.150** |
| 20k | 0.701 | **0.719** | 0.159 | **0.152** |

LPIPS aux is strictly better at every checkpoint (after step 1k where SSIM is
within noise) and **visually much better** — the qualitative jump from
"source + mild palette shift" to actual anime stylization that the step-2k
panel previewed translates through to convergence. **LPIPS aux is now
load-bearing in the recipe**; runs without it are no longer interesting.

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

### exp09 — exp08-lpips + pixel_shuffle (encoder on, mc=56) — DONE

Tests sub-pixel conv upsampling on the **proven** architecture (exp08-lpips:
encoder on, FM, LPIPS 0.2). Single-variable change vs exp08-lpips:
`--upsample-type pixel_shuffle` instead of resize+conv. ICNR init applied to
avoid checkerboard artifacts at training start.

Param-matched to exp08-lpips by lowering `--model-ch 64 → 56`, since
PixelShuffle's `channels → 4·channels` upsamplers add ~21M params. At mc=56
with pixel_shuffle the total is **44.1M / 32.9M trainable**, within ~3% of
exp08-lpips's 42.7M / 31.6M.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --freeze-source-encoder all --model-ch 56 `
    --upsample-type pixel_shuffle `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp09_pixshuf_lpips_mc56_20k `
    --wandb-tags "flow,encoder-on,lpips,pixel-shuffle,exp09" `
    --outdir out/exp09_pixshuf_lpips_mc56_20k
```

Final val curve vs exp08-lpips (resize_conv, mc=64) at every checkpoint:

| step | exp08-lpips SSIM | **exp09 SSIM** | exp08-lpips LPIPS | **exp09 LPIPS** |
|---:|---:|---:|---:|---:|
|  1k | 0.608 | **0.583** | 0.161 | **0.170** |
|  5k | 0.678 | **0.655** | 0.148 | **0.153** |
| 10k | 0.703 | **0.685** | 0.148 | **0.153** |
| 15k | 0.715 | **0.700** | 0.150 | **0.154** |
| 20k | 0.719 | **0.707** | 0.152 | **0.155** |

**Result: pixel_shuffle strictly worse on both metrics at every step.** Gap
is small (~1.5% on SSIM, ~2% on LPIPS) but persistent across the whole run.
Caveat: the param match wasn't perfect — exp09 had 32.9M trainable
(vs exp08-lpips's 31.6M), so exp09 was slightly *larger* and still lost.
The "smaller model lost" explanation is ruled out.

What this tells us:
- The fastai-era PixelShuffle intuition doesn't carry. Probably because
  LPIPS aux 0.2 is already pushing for perceptual sharpness; the marginal
  contribution of "learnable single-step upsample" disappears.
- The 4× param cost in PixelShuffle's upsamplers got reallocated to width
  (64 → 56). Wider channels at every level beat sharper upsamplers at lower
  width on this task.
- ICNR init worked: losses are smooth, no checkerboard catastrophe. The
  implementation is fine — the technique just doesn't pay off here.

**Decision: resize_conv stays the default.** Flag remains available
(`--upsample-type pixel_shuffle`) for future tasks where the trade-off may
flip (e.g. higher resolution, no aux loss).

### exp10 — multi-scale attention + bf16 on top of exp08-noenc — DONE

Hypothesis: multi-scale self-attention can replace the perceptual priors that
exp08-noenc lost when we dropped the ImageNet encoder. If true, we get a
~24M-trainable model (vs exp08-lpips's 32M+11M frozen) with equal or better
LPIPS — a real architectural win for size.

Builds on exp08-noenc (no encoder, mc=88, FM, dropout, LPIPS 0.2). Two new
variables vs exp08-noenc:
- `--attn-resolutions "8,16,32"` adds self-attention at 16x16 and 32x32
  (was bottleneck-only at 8x8). Resolution-conditional construction means
  old checkpoints still load.
- `--amp bf16` enables Tensor Core / FlashAttention path on the 4090.
  Same exp range as fp32, no GradScaler needed. ~20% step speedup +
  ~10% activation memory savings. Free win baseline-wide.

Footprint (measured on 4090, bs=4 at 128px):

| variant | params | step ms | peak VRAM |
|---|---:|---:|---:|
| exp08-lpips baseline (attn=8, fp32) | 42.7M | 39.3 | 1253 MB |
| exp08-noenc baseline (attn=8, fp32) | 23.7M | 35-ish | similar |
| **exp10 = no-enc + attn(8,16,32) + bf16, mc=88** | **24.3M** | **~30** | **~1100 MB** |

Long run, 30k steps (vs the standard 20k) since smaller model + extra aux
signal benefits from more convergence time. With bf16 step ~30% faster,
30k bf16 steps takes about the same wall-clock as 23k fp32 steps.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 30000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp10_noenc_attn832_bf16_mc88_30k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,exp10" `
    --outdir out/exp10_noenc_attn832_bf16_mc88_30k
```

Resume command shape (if killed mid-run; replace step number):

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 30000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --resume out/exp10_noenc_attn832_bf16_mc88_30k/model_step_NNNNNN.pt `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp10_noenc_attn832_bf16_mc88_30k_resumed `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,exp10,resume" `
    --outdir out/exp10_noenc_attn832_bf16_mc88_30k
```

Predictions:
- If exp10 LPIPS ≤ exp08-lpips's 0.152 → multi-scale attention can compensate
  for losing the ImageNet encoder. Smaller, simpler, encoder-free model wins.
- If exp10 LPIPS still ~0.159 (matches exp08-noenc) → attention helps but
  doesn't replace the specific ImageNet feature alignment LPIPS rewards.
  Encoder stays in the recipe.
- If exp10 SSIM > 0.734 → attention is doing real work for structure
  preservation on top of no-encoder's already-strong SSIM.

Comparison table to fill in once results land:

| step | exp08-noenc SSIM | exp10 SSIM | exp08-noenc LPIPS | exp10 LPIPS |
|---:|---:|---:|---:|---:|
| **20k** | **0.734** | **0.736** | **0.159** | **0.153** |
| **30k** | **—** | **0.740** | **—** | **0.158** |

**Step-20k snapshot — Scenario A confirmed.**

Three-way comparison at step 20k (val split, full 13 batches × bs=4 = 52 examples,
EMA + 20-step Euler):

| metric | exp08-lpips (mc=64, encoder, no attn) | exp08-noenc (mc=88, no encoder, no attn) | **exp10 (mc=88, no encoder, attn 8,16,32, bf16)** |
|---|---:|---:|---:|
| mean_loss | 0.0056 | 0.0053 | 0.0054 |
| **SSIM ↑** | 0.719 | 0.734 | **0.736** |
| **LPIPS ↓** | **0.152** | 0.159 | **0.153** |

- SSIM: exp10 beats exp08-lpips by +2.4%, ties exp08-noenc.
- LPIPS: exp10 within noise of exp08-lpips (+0.7%, well below the ~10%
  data-bound generalization gap).

**Multi-scale attention recovered the ~5% LPIPS baseline shift that
the ImageNet encoder was providing — without any external pretraining.**
exp08-noenc's LPIPS was 0.159; exp10's is 0.153, closing the encoder gap.

This is a real architectural win: encoder-free model with multi-scale
attention now matches or beats the encoder-based baseline. We can ship
a smaller, simpler, encoder-free architecture going forward. Visual
panels at step 20k show recognizable anime stylization comparable to
exp08-lpips.

**Step-30k final:**

| | exp08-lpips 20k | exp10 best (step 20k) | exp10 final (step 30k) |
|---|---:|---:|---:|
| SSIM ↑ | 0.719 | 0.736 | **0.740** (best) |
| LPIPS ↓ | **0.152** (best) | 0.153 | 0.158 |

**LPIPS regressed slightly between 20k and 30k** (0.153 → 0.158) while
SSIM kept improving. So **the optimal stopping for exp10 was ~20k**, not
30k. Likely causes: LPIPS aux signal saturated at weight 0.2 so further
training optimizes MSE/SSIM at the expense of perceptual; train-val gap
creep on the small (287-pair) dataset; cosine LR floor (1e-5) oscillation
in a flat minimum.

**Final reading**: exp10 (step 20k checkpoint) **matches exp08-lpips on
LPIPS and beats it by +2% on SSIM**, with **no encoder dependency** and
~25% smaller architecture than exp08-lpips's encoder+UNet. Architectural
win confirmed.

For exp11 onward: either cap at 20k–25k steps, or train to 30k but pick
the EMA checkpoint with best val LPIPS (we have all 1k–29k checkpoints).

### exp11 — exp10 + linear RGB — DONE (essentially neutral)

Adds physically-correct linear-RGB training on top of the confirmed-best
exp10 architecture (no encoder, mc=88, attn 8,16,32, bf16, FM, LPIPS 0.2,
dropout 0.15).

The hypothesis: FM's interpolant `x_t = (1-t)·source + t·target + σ·noise`
and bilinear upsampling are physically meaningful operations on light
intensity, which sRGB does not represent linearly. Operating in linear RGB
should clean up off-path drift and slightly sharpen edges/colors. We expect
2-5% LPIPS improvement.

Wired via `--color-space linear_rgb`. Conversions happen at boundaries:
- **Dataset**: PIL loads sRGB, conversion to linear after PIL augmentation
  ([dataset.py L160-L166](../src/img2img/dataset.py#L160-L166)).
- **Source encoder** (when used): linear → sRGB before the ResNet18 forward
  ([model.py L344-L347](../src/img2img/model.py#L344-L347)). exp11 uses
  `--no-source-encoder` so this branch is inactive but kept correct.
- **LPIPS aux**: trainer wraps the LPIPS network so it always sees sRGB
  inputs.
- **Panels**: trainer / validate.py / infer.py convert linear → sRGB at
  the display boundary so panels are display-correct and metrics are
  comparable across runs.

Validation always reports SSIM/LPIPS in sRGB regardless of training color
space, so exp11 numbers compare directly to exp08-lpips/exp10/etc.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 30000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --color-space linear_rgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp11_linrgb_noenc_attn832_bf16_mc88_30k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,linear-rgb,exp11" `
    --outdir out/exp11_linrgb_noenc_attn832_bf16_mc88_30k
```

Notes:
- `--num-workers 8` enables the full dataloader speedup (pin_memory +
  persistent_workers + non_blocking auto-on, ~2.7× data-loading throughput).
- `--checkpoint-every 5000` saves 5 intermediate + 1 final = 6 checkpoints
  total (~2 GB disk) instead of 30 (~20 GB). Still enough granularity to
  identify best-LPIPS step from the val curve. If you need finer for some
  reason, set to 2500.

Predictions (made before the run):
- LPIPS: -2 to -5% vs exp10 (0.146-0.150 range vs exp10's 0.153)
- SSIM: tiny shift, probably +/-1%.
- Visual: cleaner palette in mid-trajectory frames.

**Actual val curve:**

| step | exp10 SSIM | exp11 SSIM | exp10 LPIPS | exp11 LPIPS |
|---:|---:|---:|---:|---:|
|  5k | — | 0.686 | — | 0.160 |
| 10k | — | 0.713 | — | 0.154 |
| 15k | — | 0.725 | — | 0.153 |
| 20k | 0.736 | 0.732 | 0.153 | 0.154 |
| 30k | 0.740 | 0.737 | 0.158 | 0.156 |

**Outcome: linear RGB is essentially neutral.** Within val-set noise (~1-2%
with n=50). The -2 to -5% LPIPS prediction was wrong.

**Why linear RGB didn't deliver:**
- LPIPS aux is computed in sRGB (we convert at the boundary), so the
  perceptual gradient signal is identical between sRGB and linear RGB
  trainings. Only the MSE/v_target and bilinear-upsample paths use
  physically correct math, and at 128px those don't dominate.
- bf16 numerical noise probably drowns out small precision gains from
  linear math.
- PIL augmentations still happen in sRGB before conversion, so part of
  the data path retains sRGB nonlinearity.
- Most importantly: **all three architectures (exp08-lpips, exp10, exp11)
  converge to the same ~0.152-0.153 best-LPIPS ceiling. The architectural
  ceiling for this dataset is real.**

### Architectural ceiling — strong evidence (at 128px specifically)

| model | best LPIPS | best SSIM | trainable params | external priors |
|---|---:|---:|---:|---|
| exp08-lpips | **0.152** | 0.719 | 31.6M | ImageNet ResNet18 |
| exp10 | 0.153 | **0.736** | 43.9M | none |
| exp11 (linear) | 0.153 | 0.732 | 43.9M | none |
| floor (128px) | 0.199 | 0.617 | 0 | — |

All three architectures land within ~0.001 LPIPS of each other at 128px.
**More architecture work at 128px produces ~zero perceptual gain.**

**Caveat (added 2026-05-10 after exp12):** this ceiling turned out to be
**a 128px artifact, not an absolute one.** exp12 at 256px hit LPIPS 0.142,
substantially below the 0.152 ceiling — and the LPIPS-vs-floor improvement
went from 23% to 53%, a 2.3× relative jump. The architectures weren't out
of expressiveness; they were out of pixels. Resolution turned out to be
the highest-leverage single axis we tested.

### Recommended next moves (revised after exp11)

| candidate | expected LPIPS gain | cost | priority |
|---|---|---|---|
| **Larger paired dataset (1k-10k pairs)** | **5-10%** | data generation | **🔴 high** |
| Higher resolution (256px) | unclear, possibly meaningful | 4x compute | 🟡 medium |
| VGG-backbone LPIPS aux | unclear, plausible | one flag | 🟡 medium |
| Longer training + best-LPIPS early-stop | 1-2% | trivial | 🟡 medium |
| `--lpips-weight 0.4` ablation | 1-2% | 1 flag | 🟡 medium |
| FiLM + 2 ResBlocks (Path C continuation) | <2% | hour | 🟢 low |
| linear RGB everywhere (incl. PIL aug rewrite) | <2% | hours | ⚫ skip |

---

## Planned next experiments

(Below the line: experiments specified but not yet run. Each is a
single-variable change vs the current best architecture (exp10) so
attribution stays clean. exp12 and exp13 are orthogonal axes and can
run in parallel on the two VMs.)

### exp12 — 256px on the existing 287-pair dataset — DONE

Result table (val split, 13 batches × bs=4 = 52 examples, EMA + 20-step Euler):

| step | exp10 @ 128 SSIM | exp12 @ 256 SSIM | exp10 @ 128 LPIPS | exp12 @ 256 LPIPS |
|---:|---:|---:|---:|---:|
|  5k | — | 0.627 | — | 0.154 |
| 10k | — | 0.655 | — | 0.144 |
| 15k | — | 0.666 | — | 0.141 |
| 20k | 0.736 | 0.669 | 0.153 | **0.142** |

**Resolution-aware comparison (relative to do-nothing floor):**

| | floor | best | improvement vs floor |
|---|---:|---:|---:|
| exp10 @ 128px LPIPS | 0.199 | 0.153 | **−23%** |
| exp12 @ 256px LPIPS | 0.299 | **0.142** | **−53%** |
| exp10 @ 128px SSIM | 0.617 | 0.736 | +19% |
| exp12 @ 256px SSIM | 0.557 | 0.669 | +20% |

**Resolution more than 2× the LPIPS-vs-floor improvement** (23% → 53%).
This is by far the biggest single-axis improvement we've seen. SSIM-vs-floor
stayed roughly the same — confirming SSIM is mostly capacity-limited even at
low resolution, while LPIPS captures detail/perceptual signal that scales
with pixel budget.

LPIPS optimum: ~15k (0.141), flat through 20k. SSIM still climbing at 20k —
suggests exp14 (with more data) should extend training past 20k.

**Headline finding revision:** the "architectural ceiling at LPIPS ≈ 0.152"
we observed across exp08-lpips / exp10 / exp11 was specifically a **128px
artifact**, not an absolute ceiling. The architectures didn't run out of
expressiveness — they ran out of pixels. exp12 at 256px breaks past it
cleanly, and exp14 (with 1k pairs at 256px) should push further.

### exp14v2 — exp10 architecture on the 1k-pair dataset, 256px — DONE

After the LPIPS-aux bug fix (see Known bugs), exp14 was re-launched as
exp14v2 with the same spec on the 1k-pair dataset at 256px. Validation on
the 1k val split:

| step | LPIPS ↓ | SSIM ↑ |
|---:|---:|---:|
|  5k | 0.1770 | 0.632 |
| **10k** | **0.1767 (best)** | 0.654 |
| 15k | 0.1798 | 0.666 |
| 20k | 0.1819 | 0.674 |
| 25k | 0.1805 | 0.679 |
| 30k | 0.1816 | 0.683 |
| 35k | 0.1830 | 0.685 |
| 40k | 0.1832 | **0.686 (final)** |

**Same LPIPS-regression-while-SSIM-keeps-climbing pattern** we saw in
exp10/exp12 — LPIPS bottoms at step 10k; training past 10k trades
perceptual quality for structural fidelity. Best-LPIPS checkpoint is
step 10k, not the 40k final.

Comparing on the same 1k val set (eval on identical val pairs even for
older checkpoints):

| model | LPIPS-squeeze | SSIM |
|---|---:|---:|
| exp12 (287 pairs, 20k, 256px) | 0.190 | 0.635 |
| exp14v2 (1k pairs, 40k, 256px) | 0.183 | **0.686** |
| Δ | -3.7% | **+8.0%** |

**Surprise**: 3.5× more paired data and 2× more steps mainly bought us
**SSIM (structural diversity)**, not LPIPS (perceptual quality). This is
the opposite of what I predicted earlier. The 1k dataset has more pose /
composition variety, which exp14v2 captures (SSIM up), but the perceptual
ceiling didn't move much from data alone.

### exp15 — exp14v2 + VGG feature loss (content L1 + Gram-matrix style L1) — DONE

Adds VGG content + Gram style loss on top of exp14v2's config (1k pairs,
256px). 20k steps. Same other flags.

Final result vs baselines (same 1k val set):

| model | LPIPS-squeeze ↓ | SSIM ↑ |
|---|---:|---:|
| exp12 (287 pairs, no VGG) | 0.190 | 0.635 |
| exp14v2 (1k pairs, no VGG, 40k) | 0.183 | **0.686** |
| **exp15 (1k pairs, +VGG content+Gram, 20k)** | **0.162** | 0.631 |

- **LPIPS-squeeze: -11.5% vs exp14v2.** The biggest single-feature win
  since the resolution bump.
- **SSIM: -8.0% vs exp14v2.** Classic style-vs-structure tradeoff. VGG/
  Gram pushes the model toward texture-statistical matching at small
  cost to pixel-aligned structure. For a stylization task this is the
  right direction, but worth flagging.
- **Spike at ~step 5k**: a single bad-batch loss spike disrupted
  training. The optimizer eventually recovered and the run finished, but
  the final result is likely a couple of percent worse than it would
  have been without the corruption. The new `--max-loss-spike-ratio`
  safeguard (added 2026-05-11) skips this kind of step entirely; runs
  from here on should be more stable.

### Loss-redundancy analysis (added 2026-05-11)

`--lpips-weight 0.2` (SqueezeNet backbone, BAPPS-learned weights) and
`--feature-content-weight 1.0` (VGG16 backbone, L1 distance) are both
"pretrained-feature-distance" signals. They overlap conceptually:

| loss | backbone | weights | distance |
|---|---|---|---|
| LPIPS-squeeze | SqueezeNet (~0.7M) | BAPPS-learned | L2 of (pred - tgt) features |
| VGG content | VGG16 (~14M) | unweighted | L1 of (pred - tgt) features |
| **Gram style** | VGG16 | unweighted | L1 of channel-correlation matrices |

The Gram style term is genuinely orthogonal — it captures texture
statistics independent of pixel position. The content L1 and LPIPS are
plausibly redundant; VGG-content with weight 1.0 dominates the gradient
budget vs LPIPS-squeeze with weight 0.2.

Plus: **we're partially overfitting to metric.** Reporting LPIPS-squeeze
as the win metric while training with LPIPS-squeeze in the loss creates
a confound. Validated on exp08-noenc:

```
mean_lpips_squeeze = 0.163  (in training loop)
mean_lpips_vgg     = 0.209  (never in any training loss)
```

About 28% gap between in-loop and out-of-loop perceptual metrics on the
same model. Going forward, validate.py reports both:
- `mean_lpips_squeeze_sampled` — continuity with exp01-15
- `mean_lpips_vgg_sampled` — honest, never-in-loop perceptual check

Older results above are reported on `_squeeze` only (the metric they
were validated with at the time). Re-running validation on those
checkpoints with the updated `validate.py` will populate the `_vgg`
column too.

### exp18 — exp15 with 2× model capacity (mc=88 → 128) — PLANNED

Capacity scaling test. exp15 hit LPIPS 0.162 with the 1k dataset at
mc=88 (~44M params). The question: is that the data ceiling or the
architecture ceiling? Doubling model capacity isolates the test.

Single-variable change vs exp15: `--model-ch 128` instead of 88.

| | exp15 (mc=88) | **exp18 (mc=128)** |
|---|---:|---:|
| total params | 44.9M | **93.6M** (2.08×) |
| step time (256px, bs=4, bf16) | 255 ms | 332 ms (+30%) |
| peak VRAM | 10.0 GB | 11.3 GB |
| 20k wall-clock | ~85 min | ~110 min |

VRAM fits at bs=4 on a 16GB 4090, so no effective-batch confound — clean
single-variable test.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 128 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --feature-content-weight 1.0 --feature-style-weight 5000.0 `
    --feature-loss-layers "8,15,22" `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp18_capacity2x_mc128_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,feature-loss,2x-capacity,256px,exp18" `
    --outdir out/exp18_capacity2x_mc128_1k_256px_20k
```

Predictions:
- If exp18 LPIPS ≤ 0.155 → capacity was partially limiting. Worth going
  to 4× (exp19, mc=176, bs=2 with effective-batch confound).
- If exp18 ≈ 0.162 → architecture's not the bottleneck at this scale.
  Stop scaling capacity, look at data variety / loss design.
- Visual check is the better signal for the "shape simplicity" concern
  — doubled capacity might let the model encode more complex shapes
  even if the LPIPS metric doesn't move much.

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

### exp17_v2 — exp17 with safeguards (clean retry) — SKIPPED

Originally planned to re-run exp17 cleanly without the spike. **Skipping**
because the direction is already confirmed wrong: exp17 (VGG content+Gram
only) is worst across all metrics, and even a clean ~5% improvement
wouldn't flip the ordering vs exp14v2 or exp15. Not worth the GPU.

Same as exp15 with `--lpips-weight 0` to drop the LPIPS-squeeze aux
entirely. Tests whether SqueezeNet was contributing anything beyond
redundancy with VGG-content.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.0 `
    --feature-content-weight 1.0 --feature-style-weight 5000.0 `
    --feature-loss-layers "8,15,22" `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp17_no_lpips_only_vgg_1k_256px_20k `
    --wandb-tags "flow,no-encoder,feature-loss,no-lpips,256px,exp17" `
    --outdir out/exp17_no_lpips_only_vgg_1k_256px_20k
```

Predictions:
- If exp17 LPIPS-vgg ≈ exp15 LPIPS-vgg → LPIPS-squeeze was redundant. Drop
  it from the default recipe; saves ~3 ms/step.
- If exp17 LPIPS-vgg meaningfully worse → SqueezeNet's BAPPS weights were
  carrying signal VGG-content doesn't, despite the smaller backbone.
- Either way, exp17's LPIPS-squeeze will probably worsen slightly since
  we removed the direct training term for it.

### exp12 — original spec (kept for reference)

After exp10 / exp11 confirmed an architectural ceiling at 128px (LPIPS ~0.152
across three different architectures), the natural next axis is resolution
rather than more architecture. Higher resolution gives:

- More pixels = more LPIPS sensitivity (the metric should resolve smaller
  differences, exposing whether our models still have headroom).
- Larger receptive-field demands at the same fractional levels (more work
  for attention).
- A fairer test of how the architecture scales for the upcoming 1k-pair
  dataset (which is being generated at 1024px anyway, so we'll downscale).

Same architecture as exp10 (no encoder, mc=88, FM, LPIPS 0.2, dropout,
bf16, dataloader perf). One change: `--image-size 256` and matching
`--attn-resolutions "16,32,64"` to preserve fractional attention coverage.

| level | feat at 128px | feat at 256px | exp10 attn (8,16,32) | exp12 attn (16,32,64) |
|---|---:|---:|---|---|
| h1 | 128 | 256 | — | — |
| h2 | 64 | 128 | — | — |
| h3 | 32 | 64 | ✓ | ✓ |
| h4 | 16 | 32 | ✓ | ✓ |
| bottleneck (mid_attn) | 8 | 16 | ✓ always-on | ✓ always-on |

Cost estimates (extrapolated from 128px bench, no fresh measurement
because the GPU is busy generating the 1k dataset):

- Step time: ~120-160 ms (vs 31 ms at 128px).
- VRAM: ~5-6 GB peak (vs 1.3 GB at 128px). Comfortable on a 16 GB 4090.
- 20k steps wall-clock: ~45 min with dataloader + bf16.

Color space: keep `srgb` for now. Linear RGB was a wash at 128px; revisit
when EXR data lands.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp12_256px_noenc_attn16_32_64_bf16_mc88_20k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,256px,exp12" `
    --outdir out/exp12_256px_noenc_attn16_32_64_bf16_mc88_20k
```

If VRAM gets tight, drop `--batch-size` from 4 to 2 (default in trainer is 4).

Predictions:
- LPIPS at 256px should be **lower in absolute terms** because the metric
  has more spatial bandwidth to penalize errors. Our 128px LPIPS of 0.152
  doesn't translate directly — what matters is the relative gap to the
  256px floor, which we'd want to also compute.
- Visual quality should be noticeably better — anime-style flat regions
  and edges show up more crisply at 256.
- Memorisation risk: same 287 pairs, more pixel content per pair, so it's
  unclear if data scarcity dominates more or less at this resolution.
  Train-val gap diagnostic worth running again after this finishes.

### exp13 — exp10 + VGG-backbone LPIPS aux — PLANNED

Single-flag follow-up. Switches the LPIPS auxiliary loss backbone from
SqueezeNet (~700k params) to VGG16 (~14M params).

Motivation:
- All three architectures tested at 128px (exp08-lpips, exp10, exp11) hit
  the same LPIPS ceiling around 0.152-0.153. One plausible cause: the
  SqueezeNet aux loss is too crude to drive the model past a certain
  perceptual quality floor.
- The classic style-transfer literature (Gatys, Johnson, AnimeGAN) all uses
  VGG features specifically because deeper VGG layers (`relu4_*`, `relu5_*`)
  encode texture/style signal well — which is exactly what anime stylization
  is about.
- LPIPS-VGG retains LPIPS's learned reweighting from the BAPPS dataset on
  top of the larger VGG backbone.

Cost: ~10 ms/step extra at bs=4/128px (vs ~2 ms for SqueezeNet). ~3 min
extra total wall-clock over 20k steps. Negligible.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.2 --lpips-aux-net vgg `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp13_vgg_lpips_noenc_attn832_bf16_mc88_20k `
    --wandb-tags "flow,no-encoder,lpips-vgg,attn-multiscale,bf16,exp13" `
    --outdir out/exp13_vgg_lpips_noenc_attn832_bf16_mc88_20k
```

Note: the **validation metric stays on LPIPS-SqueezeNet** for continuity
with exp01-exp11 numbers. exp13 results will be directly comparable to
exp10. If exp13 improves on the same SqueezeNet-LPIPS metric, the bigger
VGG aux loss was the bottleneck.

Predictions:
- If LPIPS-squeeze improves > 3% vs exp10 → VGG features were a real
  bottleneck for our perceptual ceiling. Adopt VGG as the new default
  aux loss for the 1k-pair dataset run too.
- If LPIPS-squeeze ≈ exp10 → the ceiling is data-bound, not aux-loss-bound.
  Stay with SqueezeNet (cheaper) and focus on data.

### exp15 — exp10 + VGG feature loss (content L1 + Gram-matrix style L1) — PLANNED

Adds fastai-style `FeatureLoss` (Gatys / Johnson texture-aware loss) on
top of the exp10 architecture. Where LPIPS measures "do these two images
look similar?", Gram-matrix style loss measures "do these two images
have the same texture statistics?" — which is what stylization actually
demands. Anime stylization is fundamentally a texture transformation
(photo skin/hair textures → anime flat regions + ink lines), so this is
a plausibly high-leverage axis.

[src/img2img/feature_loss.py](../src/img2img/feature_loss.py) implements:
- Frozen VGG16 forward at chosen layers (default: 8, 15, 22 = after
  relu2_2 / relu3_3 / relu4_3, the fastai/Johnson convention).
- **Content L1**: feature distance at each layer.
- **Style L1**: Gram-matrix distance at each layer. Gram is the
  channel-wise covariance, capturing texture statistics independent of
  spatial position.
- ImageNet mean/std normalisation before VGG; input clamped to [0, 1].
- For `--color-space linear_rgb`, converts to sRGB at the boundary.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --feature-content-weight 1.0 --feature-style-weight 5000.0 `
    --feature-loss-layers "8,15,22" `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp15_featureloss_noenc_attn832_bf16_mc88_20k `
    --wandb-tags "flow,no-encoder,lpips,feature-loss,gram-style,bf16,exp15" `
    --outdir out/exp15_featureloss_noenc_attn832_bf16_mc88_20k
```

**Loss magnitude warning**: `style_weight=5000` (fastai/Johnson default)
makes the style term dominate the total loss. Smoke test at step 30
showed total loss ~10-30 vs diffusion ~0.02 and lpips ~0.05. **This is
expected** — the absolute total isn't directly comparable to other
exp totals. What matters is the per-term values logged separately
(`train/feature_content`, `train/feature_style`) and the **val LPIPS**
on the standard SqueezeNet metric.

Tuning candidates if results look off:
- Style dominating too much → drop to `--feature-style-weight 1000` or `500`.
- Texture matching too aggressive (output looks "over-stylized") → lower style weight.
- Texture not transferring (output stays photo-like) → raise style weight or add deeper VGG layers (e.g. `--feature-loss-layers "8,15,22,29"`).
- **Per-layer weighting** (added 2026-05-11): currently we average per-layer
  losses uniformly. fastai's recipe used `[5, 15, 2]` for the three default
  layers (relu2_2 / relu3_3 / relu4_3) to emphasize mid-level texture/edge
  features over the deepest semantic layer. Enable via:
  `--feature-content-layer-weights "5,15,2" --feature-style-layer-weights "5,15,2"`.
  Note: weights are raw multipliers, summed not averaged. fastai's weights
  sum to 22 (vs uniform 1.0), so the effective content/style magnitude is
  ~22× higher; compensate by lowering `--feature-content-weight` and
  `--feature-style-weight` to ~1/22 of current values OR accept the larger
  magnitude (the safeguards will skip steps if it spikes).

Predictions:
- If exp15 val LPIPS-squeeze < exp10's 0.153 by > 3% → Gram texture
  matching was a real bottleneck. Texture-aware loss is the way forward
  for stylization tasks.
- If exp15 ≈ exp10 → LPIPS already captured what Gram does for our
  task (unlikely but possible). Stay with simpler LPIPS aux.
- Visual check is more informative than the LPIPS number here — the
  expected gain is "stronger anime stylization" which may not score on
  pairwise LPIPS but is the qualitative win.

### exp19 — exp14v2 + `--lpips-weight 0.4` (double LPIPS aux) — PLANNED

After the fair-step revalidation showed exp14v2 (LPIPS only) is the
strongest recipe, the natural follow-up is to lean harder on the signal
that's working. Single-flag change: double the LPIPS aux weight from 0.2
to 0.4.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.4 `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp19_lpips04_noenc_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,lpips-0.4,bf16,1k-dataset,256px,exp19" `
    --outdir out/exp19_lpips04_noenc_1k_256px_20k
```

Predictions:
- If exp19 lpips_vgg < 0.248 (exp14v2 at 20k) → LPIPS weight was
  under-tuned. Adopt 0.4 as new default.
- If exp19 ≈ exp14v2 → LPIPS at 0.2 was already saturating. Stay there.
- If exp19 worse → 0.4 over-emphasizes perceptual at cost of MSE/structure.
  Bracket the optimum: try 0.3 next.

### exp22 — exp14v2 + `--lpips-aux-net vgg` (LPIPS backbone swap) — PLANNED

The other axis to test on the winning recipe: does LPIPS-VGG beat
LPIPS-squeeze when used as the *only* perceptual aux (no Gram, no
content-L1 on top)? This is a cleaner version of what exp13 was
originally supposed to do, but now against a known-good baseline.

Single-flag change from exp14v2: `--lpips-aux-net vgg` instead of squeeze.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 --lpips-aux-net vgg `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp22_lpips_vgg_noenc_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips-vgg,bf16,1k-dataset,256px,exp22" `
    --outdir out/exp22_lpips_vgg_noenc_1k_256px_20k
```

Notes:
- We've now confirmed lpips_vgg is the honest metric (not in any training
  loop here either, since we'd train against LPIPS-VGG features but the
  metric uses different layers/weights).
- Adds ~10ms/step (~3 min extra over 20k).
- LPIPS-squeeze metric will likely worsen because we removed it from the
  loss; lpips_vgg metric is the one that matters for this comparison.

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

### exp20 — vanilla GAN aux (no NoGAN phasing) — DONE, grid-artifact collapse

First attempt at adding the GAN aux on top of the exp14v2 recipe, with
**both G and D updating from step 1** (no pretrain phasing). Used the
PatchGAN discriminator + hinge loss + spectral norm setup described
above, at `--gan-weight 0.1`.

**Result: training "succeeded" by loss curves but visually produced
grid-like artifacts** — periodic patterns roughly at the PatchGAN's
~70-pixel receptive-field scale. The G output composed locally
plausible 70×70 patches that didn't tile into coherent images. Classic
GAN-from-scratch failure mode: random D produces meaningless gradient
direction in the early steps, G wanders into a degenerate "fool the
discriminator per patch" mode before learning the basic photo→anime
mapping.

This is *exactly* what fastai's NoGAN three-phase approach was designed
to prevent. Implemented and queued as exp21.

### exp21 — GAN aux with fastai NoGAN three-phase scheduling — IMPLEMENTED, PLANNED to launch

**Status update (2026-05-11)**: now implemented and smoke-tested. The
exp15/exp16/exp17 sequence revealed that VGG content + Gram is NOT what
made fastai's image generation work — the **GAN adversarial loss** was the
load-bearing piece. We now have a clean test: add GAN to the proven
exp14v2 recipe (LPIPS only, no VGG) and see if it closes the shape-
simplicity gap.

**Implementation** (~250 lines across three new files):
- [src/img2img/discriminator.py](../src/img2img/discriminator.py):
  PatchGAN discriminator, pix2pix 70×70 receptive field at default depth.
  Spectral norm on every conv. ~2.8M params at `--gan-d-channels 64`.
- [src/img2img/gan_loss.py](../src/img2img/gan_loss.py): hinge GAN losses
  (`hinge_d_loss`, `hinge_g_loss`). Hinge pairs naturally with spectral
  norm and avoids BCE saturation.
- [experiments/010_img2img_photo2comics/train.py](../experiments/010_img2img_photo2comics/train.py):
  GAN flags, separate AdamW for D (lr=1e-4, β1=0.5 by pix2pix
  convention), G update with adversarial term, D update on detached
  fake. Discriminator state + opt_d state saved in checkpoints (preserved
  on `--resume`). wandb logs `train/g_gan_loss`, `train/d_loss`,
  `train/d_real_score`, `train/d_fake_score`.

Smoke-tested at `--gan-weight 0.1`: D loss settles to its 2.0 hinge
equilibrium, real and fake scores track each other (D learning to
discriminate fairly), no NaN or spike under bf16 autocast.

**NoGAN phase scheduling (added 2026-05-11)**: implementing fastai's three-
phase approach to avoid random-G-meets-random-D chaos. Two new flags:

- `--gan-pretrain-g-steps N`: phase 1, GAN inactive. G trains on
  LPIPS/feature only. Reaches a "reasonable" baseline.
- `--gan-pretrain-d-steps M`: phase 2, G frozen. D trains alone on
  `(real, current-G-output)` pairs. Calibrates D before adversarial play.
- After `N + M` steps, phase 3 (full GAN) starts.

The smoke test confirms all three phases activate correctly at their
step boundaries and the metric printout shows the active phase
(`phase g_pretrain / d_pretrain / full`).

```powershell
# Recommended exp21 launch with NoGAN phasing.
# Phase 1 (5k): pure LPIPS pretrain  — G learns the basics
# Phase 2 (2k): D calibration       — D catches up on current G's output
# Phase 3 (13k): full adversarial   — alternating G+D updates
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --gan-weight 0.1 --gan-d-channels 64 --gan-d-layers 3 --gan-d-lr 1e-4 --gan-d-beta1 0.5 `
    --gan-pretrain-g-steps 5000 --gan-pretrain-d-steps 2000 `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp21_nogan_lpips_noenc_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,gan,patch-gan,nogan-phased,bf16,1k-dataset,256px,exp21" `
    --outdir out/exp21_nogan_lpips_noenc_1k_256px_20k
```

**Alternative: resume from exp14v2's pretrained G.** If we want to skip
phase 1 (since exp14v2's checkpoint is already an LPIPS-pretrained G),
we can launch with `--resume out/exp14v2_*/model.pt --gan-pretrain-g-steps 0
--gan-pretrain-d-steps 2000 --steps 22000`. But: exp14v2's cosine LR is
already at lr_min (1e-5) at step 40k, so the resume must override the
LR schedule (drop `--lr-cosine`, set `--lr 1e-4 --lr-warmup-steps 100`)
to actually train. Worth trying as a follow-up if the from-scratch exp21
is promising.

**Why this is now the highest-priority experiment:**

After exp17/exp16 confirmed that VGG content + Gram alone doesn't beat
LPIPS-only, the "fastai had a richer aux loss" hypothesis collapses.
What's left is the **discriminator** — which directly attacks the
shape-simplicity failure mode by saying "this output looks too simple
to be real" in a learnable, per-image way that no static perceptual
loss can replicate. This is exactly what the pix2pix lineage (pix2pix
→ AnimeGAN → AnimeGANv3) relies on for crisp stylization quality.

**Tuning notes for the launch:**
- Start with `--gan-weight 0.1` (auxiliary regularizer, not primary
  signal). If outputs look unchanged from exp14v2, raise to 0.3 or 0.5.
  If outputs look unstable / artifact-laden, lower to 0.05 or kill.
- Watch `d_real_score` and `d_fake_score` in wandb. Healthy training:
  both drift toward small positives/negatives but stay within ~±2.
  If `d_real_score` → +1.0 and `d_fake_score` → -1.0 with no movement,
  D has won (no useful G gradient). If they oscillate wildly, λ_gan
  is too high.
- D updates happen every step (1:1 ratio with G). If D is winning too
  hard later, we could go 1:2 (D every other step) — currently not
  exposed as a flag, but easy to add.

Predictions:
- If exp21 visually shows crisper shapes / more detail than exp14v2 →
  the "GAN drives shape complexity" hypothesis holds, *and* the NoGAN
  phasing was what fixed the grid-artifact failure of vanilla exp20.
  exp21 becomes the new recipe.
- If exp21 ≈ exp14v2 visually + slight metric improvement → GAN helps
  but is not transformative. Worth keeping at low weight.
- If exp21 hurts metrics or shows training instability → either λ_gan
  is mis-tuned (drop to 0.05), or NoGAN phasing alone wasn't enough
  to stabilise (would need longer phase 2 or smaller D).
- If exp21 *also* shows grid artifacts → PatchGAN's receptive field
  itself is the culprit. Try `--gan-d-layers 2` (smaller receptive
  field, ~34×34 instead of 70×70) or revisit the GAN approach entirely.

#### Original vanilla-GAN exp20 spec preserved below (for diff vs the NoGAN-phased exp21 launch above)

Triggered by visual inspection finding **shape simplicity** in our outputs:
crisp lines where they exist, but fewer of them than the target. The model
draws "soft cartoon" — locally plausible but topologically simpler than the
target anime art.

Distinct from blur (positional uncertainty smearing pixels). Shape
simplicity is about the model finding a *low-complexity attractor*: the
common subset of shapes across pairs that are correct most often. This is
what GANs in the pix2pix lineage (pix2pix → pix2pixHD → AnimeGAN →
AnimeGANv3) explicitly fix: a discriminator can identify
"this is too smoothed/simple to be real anime" in ways static perceptual
losses can't.

**Why our other losses don't fully fix this:**
- LPIPS at weight 0.2 is a regression-to-feature-mean tilt that *encourages*
  simplification.
- Gram (exp15) matches statistical complexity but doesn't punish per-image
  simplification — two images with similar Gram can have different complexity.
- L1/MSE rewards average shapes when uncertain.

**Architecture sketch:**
- Small PatchGAN discriminator (à la pix2pix), ~3M params (vs our 24M
  generator). Takes `(source, output)` pair, outputs per-patch real/fake
  scores.
- Spectral norm on every discriminator conv (already shipped as
  `torch.nn.utils.spectral_norm`; the standard pix2pix recipe).
- **Hinge loss** preferred over BCE: more stable, avoids the
  discriminator-saturation issue.
- λ_gan = 0.1 → 1.0; start at 0.1 (auxiliary regularizer, not primary
  signal). pix2pix used 1.0 with bigger discriminators; AnimeGAN uses
  ~0.5-1.0.
- Alternating G/D updates each step (1:1) or D-every-other-step (1:2).
- Discriminator LR 1e-4 with AdamW(β1=0.5, β2=0.999) (pix2pix convention).

**Cost:**
- ~150 lines: discriminator module, GAN loss helper, alternating updates
  in trainer, separate D optimizer + EMA.
- ~30% slower per step (extra D forward+backward).
- 2-3 short runs to tune λ_gan and D update frequency.
- Stability risk: GANs can collapse. Mitigated by spectral norm + small λ.

**Why we didn't implement it yet:**
- exp12 (256px) and exp14 (1k pairs) are likely to address most of the
  shape-simplicity issue at lower complexity cost.
- GAN training is real engineering and we should defer it until we know
  the simpler levers can't get there.
- This spec exists so we know what "exp20" means if we need to escalate.

```powershell
# (sketch only — implementation TBD)
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 30000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.1 `
    --gan-weight 0.1 --gan-d-channels 64 --gan-d-lr 1e-4 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-tags "flow,no-encoder,lpips,gan,exp20" `
    --wandb-run-name exp20_gan_noenc_attn832_bf16_mc88_30k `
    --outdir out/exp20_gan_noenc_attn832_bf16_mc88_30k
```

Predictions:
- If exp12 + exp14 close the shape-simplicity gap → exp16 unnecessary,
  skip the implementation cost.
- If they don't → exp16 is the standard pix2pix-lineage fix and very
  likely (~80% confidence) to give visible complexity gains. Cost
  ~150 LoC + a stability-conscious tuning pass.

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

## exp14v2 — 1k dataset, 256px, no-encoder, attn(16,32,64), bf16, mc88, LPIPS-squeeze, 40k steps

**Status: DONE 2026-05-11**

Best run so far at the time. Key results (val on 1k val set, 256px):

| step | lpips_sq | lpips_vgg | ssim  |
|------|----------|-----------|-------|
| 5k   | 0.177    | —         | 0.632 |
| 10k  | 0.177    | —         | 0.654 |
| 20k  | 0.182    | 0.248     | 0.674 |
| 40k  | 0.183    | 0.240     | 0.686 |

**Pattern**: LPIPS plateaus early (~10k), SSIM keeps climbing through 40k.
exp12 baseline on same val set: lpips_sq=0.190, ssim=0.635 → 1k dataset helps.

---

## exp15 — exp14v2 recipe + VGG content L1 + Gram style L1 (fastai FeatureLoss)

**Status: DONE 2026-05-11**

Settings: same as exp14v2 20k but `--feature-content-weight 1.0 --feature-style-weight 5000 --feature-loss-layers "8,15,22"` + `--lpips-weight 0.2` + `--source-dropout 0.15`, `attn-resolutions "8,16,32"` (not 16,32,64).

Results (20k, dual LPIPS val):
- lpips_squeeze=0.162, lpips_vgg=0.293, ssim=0.631

**Finding**: squeeze improved 11.5% vs exp14v2 20k, but VGG *degraded* (+18%).
Style-vs-structure tradeoff: Gram style loss pushes texture statistics away from
pixel-accurate targets. SqueezeNet is insensitive to this drift; VGG catches it.
The squeeze metric was overfitting to the style-transfer distribution.

---

## exp16 — fastai per-layer feature weights [5,15,2], no LPIPS

**Status: DONE 2026-05-11**

Settings: `--feature-content-weight 0.045 --feature-style-weight 227 --feature-loss-layers "8,15,22" --feature-content-layer-weights "5,15,2" --feature-style-layer-weights "5,15,2" --lpips-weight 0`, 1k dataset, 256px, attn(16,32,64).

Results: lpips_sq=0.167, lpips_vgg=0.314, ssim=0.610

**Conclusion**: fastai layer weighting doesn't fix the VGG-path problem. All
VGG content+Gram runs (exp15, exp16, exp17) lose to plain LPIPS-only on lpips_vgg
and SSIM. VGG content+Gram is not the right aux loss for this task.

---

## exp17 — VGG content+Gram only, no LPIPS

**Status: DONE 2026-05-11**

Settings: exp15 but `--lpips-weight 0`.

Results: lpips_sq=0.189, lpips_vgg=0.337, ssim=0.600

**Conclusion**: Removing LPIPS from exp15 makes everything worse. LPIPS was not
redundant — it was load-bearing. VGG feature loss alone is the worst of the three.

---

## VGG feature loss ablation summary (exp15/16/17)

| model | lpips_sq | lpips_vgg | ssim  | notes |
|-------|----------|-----------|-------|-------|
| exp14v2 20k (LPIPS-sq only) | 0.182 | 0.248 | 0.674 | baseline |
| exp15 (LPIPS-sq + VGG feat) | 0.162 | 0.293 | 0.631 | sq improved, vgg worse |
| exp16 (fastai weights, no LPIPS) | 0.167 | 0.314 | 0.610 | layer weights don't help |
| exp17 (VGG feat only) | 0.189 | 0.337 | 0.600 | worst overall |

**Verdict**: VGG content+Gram path is genuinely worse for photo→anime. The Gram
style term pushes texture statistics in directions that SqueezeNet rewards but
VGG LPIPS (the honest metric) penalises. Drop this axis.

---

## exp20 — GAN aux, gan-weight 0.1, no NoGAN phasing

**Status: KILLED 2026-05-11 (overcooked)**

D dominated from the start. At step 10k: d_real +1.5→+2.2, d_fake -1.5→-2.8.
In-loop LPIPS stuck at 0.35–0.41 (vs 0.06 healthy). G couldn't fool D at all.

**Lesson**: GAN without NoGAN phasing collapses when D is stronger than G.

---

## exp21 — GAN aux + NoGAN phases (5k G pretrain, 2k D pretrain), gan-weight 0.1

**Status: DONE 2026-05-11**

Results: lpips_sq=0.299, lpips_vgg=0.489, ssim=0.409 — catastrophic.

Phase 3 (full GAN) still had D dominating by end: d_real +1.9, d_fake -2.4,
g_gan 1.5–2.4. The NoGAN phasing helped at step 7k but D won over 13k adversarial
steps. `gan_weight=0.1` is too strong.

---

## exp21b — GAN aux + NoGAN phases, gan-weight 0.005

**Status: DONE 2026-05-11**

Results: lpips_sq=0.187, lpips_vgg=0.284, ssim=0.640

Training stable (D scores near zero), but gan-weight too weak to improve over
exp14v2. GAN at 0.005 adds adversarial noise without enough signal.

---

## exp21c — GAN aux + NoGAN phases + adaptive switching, gan-weight 0.1

**Status: DONE 2026-05-11**

Implemented `--gan-adaptive-switch`: in full phase, update G-adv if
`g_gan_ema >= d_loss_ema`, update D if `d_loss_ema >= g_gan_ema`. EMA alpha=0.1.

Results: lpips_sq=0.211, lpips_vgg=0.320, ssim=0.606

Training stable (ema_g≈0.68, ema_d≈0.89 at end — balanced). But val metrics
still worse than exp14v2. G learned to fool D on train distribution but doesn't
generalise to val. Observation: GAN helps texture/colour but loses facial detail.

**GAN conclusion**: All GAN variants (exp20/21/21b/21c) trail exp14v2 on honest
metrics. Balanced training (adaptive switch) is better than fixed phases, but
adversarial pressure at any tested weight hurts reconstruction quality vs pure
LPIPS. GAN may help qualitatively (texture, colour) even when metrics regress.
Parked pending further investigation.

---

## exp22 / exp22b — exp14v2 x4 model size (mc=176, resize_conv / pixel_shuffle)

**Status: KILLED 2026-05-11 (grid artifacts)**

mc=176 → 175M params (resize_conv) or 259M (pixel_shuffle). Both showed
repeating periodic artifacts at ~4k steps. Root cause: upsampling channels
(704→1408) too strong at this capacity, imprinting patterns.
Additionally, model converged to identity mapping (copy source) — too much
capacity for 908 training pairs at 256px.

**Lesson**: mc=176 is above the capacity ceiling for this dataset. mc=88 (44M)
is the practical limit at 1k pairs, 256px.

---

## exp23 — exp14v2 20k + LPIPS-VGG backbone (--lpips-aux-net vgg)

**Status: DONE 2026-05-11**

Single change from exp14v2: swap LPIPS training backbone from squeeze to vgg.
Val metric (squeeze) stays for continuity.

Results: lpips_sq=0.127, lpips_vgg=0.234, ssim=0.689

**Biggest single-run improvement of the project.** −30% lpips_sq vs exp14v2 20k,
−3% lpips_vgg vs exp14v2 40k, SSIM matches exp14v2 40k in only 20k steps.
VGG LPIPS as training loss forces the model to respect mid-level feature structure
(relu2_2/relu3_3/relu4_3) — exactly what matters for facial detail and edge quality.

**New baseline**: exp23 replaces exp14v2 as the reference recipe.

---

## exp24 — exp23 + native-res crops (aug-resize-scale 4.0, no zoom)

**Status: DONE 2026-05-11**

Hypothesis: downscaling 1024px→281px before cropping (default scale=1.10)
blurs stroke widths. Scale=4.0 keeps 1024px native, random crop 256px.

Results: lpips_sq=0.273, lpips_vgg=0.449, ssim=0.537 — much worse than exp23.

Root cause: 256 from 1024 = 1/16th image area per crop. High crop variance:
easy (background) crops dominate early training, then model hits hard (face)
crops at step ~4k and needs 6k steps to recover. Wasted half the training budget.

**Lesson**: native-res crops are too sparse for 256px training at 1k pairs. Need
larger effective receptive field per crop. scale=2.0 (512→256 crop, 1/4 image
area) is the planned fix (exp24b).

---

## exp24b — exp23 + scale=2.0 crops (resize 1024→512, random crop 256)

**Status: DONE 2026-05-11**

Hypothesis: scale=2.0 gives consistent stroke width (2x downscale from native)
with manageable crop variance (1/4 image area vs 1/16 for scale=4.0 or ~full
image for scale=1.10).

```bash
python3 experiments/010_img2img_photo2comics/train.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 --image-size 256 --batch-size 4 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 --lr-cosine \
    --grad-clip-norm 1.0 --no-source-encoder --source-dropout 0.0 \
    --method flow --flow-sigma-noise 0.05 --amp bf16 \
    --model-ch 88 --attn-resolutions "16,32,64" \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --aug-resize-scale 2.0 --aug-scale-jitter 0.0 \
    --sample-panel-steps 20 --checkpoint-every 5000 \
    --val-every 1000 --panel-every 1000 \
    --outdir out/exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k
```

Results: lpips_sq=0.168, lpips_vgg=0.304, ssim=0.642

**Worse than exp23** (scale=1.10). Intermediate scale (2x downscale) removes
fine stroke variation but crops still cover less structure per sample than
scale=1.10. No recovery spike seen (unlike scale=4.0), but quality ceiling is
lower. scale=1.10 (resize to ~281px, ~full image visible) remains best.

**Winner of exp23 vs exp24b**: exp23 → used as base for exp25.

---

## exp25 — exp23 recipe × 80k steps (long run)

**Status: DONE 2026-05-12**

Best recipe from ablation study (exp23: LPIPS-VGG, mc=88, attn 16/32/64, bf16,
scale=1.10, no encoder) extended to 80k steps to test continued improvement.

```bash
python3 experiments/010_img2img_photo2comics/train.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 80000 --image-size 256 --batch-size 4 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 --lr-cosine \
    --grad-clip-norm 1.0 --no-source-encoder --source-dropout 0.0 \
    --method flow --flow-sigma-noise 0.05 --amp bf16 \
    --model-ch 88 --attn-resolutions "16,32,64" \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --aug-resize-scale 1.10 --aug-scale-jitter 0.10 \
    --sample-panel-steps 20 --checkpoint-every 10000 \
    --val-every 5000 --panel-every 5000 \
    --wandb-run-name exp25_lpipsvgg_80k_from_exp23 \
    --outdir out/exp25_lpipsvgg_80k_from_exp23
```

Checkpoint progression (validated with 25 batches, 20 sample steps, EMA):

| step | lpips_sq | lpips_vgg | ssim  |
|------|----------|-----------|-------|
| 10k  | 0.133    | 0.243     | 0.671 |
| 20k  | 0.128    | 0.234     | 0.688 |
| 30k  | 0.124    | 0.228     | 0.698 |
| 40k  | 0.120    | 0.223     | 0.702 |
| 50k  | 0.117    | 0.219     | 0.708 |
| 60k  | 0.116    | 0.218     | 0.711 |
| 70k  | 0.116    | 0.218     | 0.712 |
| 80k  | 0.115    | 0.217     | 0.712 |

**Findings**: Monotonic improvement across all metrics. lpips_sq at 20k (0.128)
exactly matches exp23, confirming recipe reproducibility. Improvement rate
slows sharply after 60k: 10k→60k averages −0.005 lpips_sq per 10k steps;
60k→80k gains only −0.001 total. **Diminishing returns past 60k steps.**
Best checkpoint for deployment: 80k (final model.pt) — marginal gains justify
the full run but 60k is nearly as good. SSIM 0.712 vs 0.557 source floor.

---

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

## exp31 — exp25 fine-tune at 512px with source corruption robustness

**Status: IN PROGRESS 2026-05-13** (steps 80k→90k)

Fine-tunes the exp25 checkpoint (best single-frame model) at 512×512 with source
corruption to improve robustness to real-video blur and compression artifacts.

Architecture: identical to exp25 (flow FM, mc=88, attn 16/32/64, no source encoder,
LPIPS-VGG 0.2). All weights trainable (no freeze).

**Key changes vs exp25**:
- Resolution: 256px → 512px
- Augmentation: `aug_resize_scale=2.0` (crops 512 from ~1024px images)
- Source corruption per image (target always clean):
  - 20% chance: no corruption (clean source)
  - 80% chance: independently apply blur σ∼U[0.5,3.0] (70% prob) and/or
    JPEG quality∼U[30,95] (70% prob)
- LR: 2e-5 → 1e-6 cosine (vs 2e-4 for original exp25 — fine-tune rate)
- 10k steps (resume step 80k → target 90k)

```bash
OUTDIR=out/exp31_corrupt512_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTDIR
PYTHONPATH=/tmp/extpkgs2:/home/researcher/workspace/nanoWarp \
TORCH_HOME=/tmp/torch_home \
MPLCONFIGDIR=/tmp/mplconfig \
WANDB_API_KEY=wandb_v1_... \
WANDB_CACHE_DIR=/tmp/wandb_cache \
WANDB_CONFIG_DIR=/tmp/wandb_config \
python3 experiments/010_img2img_photo2comics/train_exp31_corrupt512.py \
    data/photo2anime_1k/photo2anime_1k \
    --resume out/exp25_lpipsvgg_80k_from_exp23/model.pt \
    --steps 10000 --image-size 512 --aug-resize-scale 2.0 \
    --lr 2e-5 --lr-min 1e-6 --lr-warmup-steps 200 \
    --corrupt-blur-max 3.0 --corrupt-jpeg-min 30 --clean-prob 0.2 \
    --wandb --wandb-run-name exp31_corrupt512 \
    --outdir $OUTDIR \
    2>&1 | tee $OUTDIR/train.log
```

Outdir: `out/exp31_corrupt512_20260513_214306/`
Wandb: https://wandb.ai/alx-spirin/nanoWarp/runs/4k1iquss

**Wandb debug**: Earlier runs failed with `CommError: user is not logged in` despite
`WANDB_API_KEY` set. Root cause: the stored API key had expired — `wandb.login()`
returns `True` without verifying the key; the Go subprocess (`wandb-core`) is what
actually calls the API and fails. Secondary issues: `~/.cache/wandb` and
`~/.config/wandb` not writable → fixed with `WANDB_CACHE_DIR` and `WANDB_CONFIG_DIR`.
Full debug notes in [captains_log_video.md#wandb-auth-failures](captains_log_video.md).

**Val curve** (clean sources, ↓ better):

| step  | lpips_sq | ssim   |
|-------|----------|--------|
| 81000 | 0.1447   | 0.6347 | ← same as exp25 (1k into fine-tune)
| 82000 | 0.1622   | 0.6161 |
| 84000 | 0.1767   | 0.6032 |
| 86000 | 0.1794   | 0.6031 |
| 88000 | 0.1853   | 0.5964 |
| 90000 | 0.1824   | 0.5973 |

**Conclusion**: clean-val degraded 0.1447 → 0.1824 (+26% LPIPS regression).
Expected — same pattern as exp30 (temporal corruption). The model learns to
de-corrupt sources, which changes its response to clean sources. For clean-source
inference use exp25 (step 80k). For real-video compressed inputs, use exp31 final
(step 90k) — nat1 nat1_step_09*.png frames will show whether it improved visually.

Nat1 frame-0 panels saved every 1k steps in the outdir.

---

## exp32 — train from scratch, progressive 128→256→512px, full augmentation

**Motivation**: exp31 showed fine-tuning from exp25 on corrupted inputs degrades clean-val
(0.1447→0.1824). Training from scratch with progressive resolution and rich augmentation
should produce a model that is simultaneously robust to real-video compression *and*
high quality on clean sources, since it never overfit to clean-only training first.

**Architecture**: identical to exp25 — mc=88, no source encoder (source in stem),
attn_res=(16,32,64), flow FM, LPIPS-VGG weight 0.2, bf16 AMP.

**Progressive phases**:

| Phase | Steps | Resolution | BS | Effective 512px-equivalent steps |
|-------|-------|------------|-----|----------------------------------|
| 1     | 5k    | 128px      | 64  | ~80k (16× area ratio)            |
| 2     | 20k   | 256px      | 16  | ~80k (4× area ratio)             |
| 3     | 75k   | 512px      | 4   | 75k                              |

**Augmentation** (per sample, randomized):

Shared geometry (source + target):
- Zoom scale ~ U[1.0, 2.5] → resize → random crop
- Rotation ±25°
- Perspective warp distortion=0.15, p=0.5
- Horizontal flip p=0.5

Source-only color jitter: brightness/contrast/saturation ±0.3

Source-only degradation (80% of samples, gated by clean_prob=0.2):
- Resize-down+up p=0.3 (factor ~ U[0.25, 0.75]) — "internet/compression" pixelation
- Gaussian blur σ~U[0.5,3.0] p=0.7
- JPEG quality~U[30,95] p=0.7

**LR**: 2e-4 → 1e-6 cosine over 100k steps, warmup 500 steps.
**Val/checkpoints**: val+panel+nat1 every 5k, checkpoint every 10k, best saved on val LPIPS.

```bash
OUTDIR=out/exp32_prog512_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTDIR
PYTHONPATH=/tmp/extpkgs2:/home/researcher/workspace/nanoWarp \
TORCH_HOME=/tmp/torch_home \
WANDB_API_KEY=wandb_v1_... \
WANDB_CACHE_DIR=/tmp/wandb_cache \
WANDB_CONFIG_DIR=/tmp/wandb_config \
MPLCONFIGDIR=/tmp/mplconfig \
python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --wandb --wandb-run-name exp32_prog512 \
    --outdir $OUTDIR \
    2>&1 | tee $OUTDIR/train.log
```

Outdir: `out/exp32_prog512_20260514_033045/`
Wandb: https://wandb.ai/alx-spirin/nanoWarp/runs/wes3p2ce

Results: TBD (run in progress, 100k steps)

---

## exp33 — exp23 recipe (20k @ 256px bs=4) with the full exp32 aug stack

**Status: RUNNING 2026-05-14**

Clean A/B vs exp23 (lpips_sq=0.127) to isolate the aug stack's impact at
fixed compute. Single architectural delta vs exp23: none — only the
augmentation pipeline changes. exp32 confounds aug × progressive-res × 100k
steps; exp33 strips out the first two confounds.

**Aug stack** (vs exp23's `scale=1.10` + hflip):
- Shared geometry: zoom scale U[1.0, 2.5], rotate ±25°, perspective 0.15
  @ p=0.5, hflip p=0.5.
- Source-only color jitter: brightness/contrast/saturation ±0.3.
- Source-only degradation (clean_prob=0.2): resize-down+up p=0.3
  (factor U[0.25, 0.75]), Gaussian blur σ U[0.5, 3.0] p=0.7,
  JPEG quality U[30, 95] p=0.7.

**Recipe** (everything else matches exp23): mc=88, attn 16/32/64, no source
encoder (source-in-stem), flow FM, LPIPS-VGG weight 0.2, bf16,
lr 2e-4 → 1e-5 cosine, warmup 500, 20k steps.

```bash
WANDB_API_KEY=... bash scripts/run_exp33_aug32stack_at_exp23_recipe.sh
```

Script: `scripts/run_exp33_aug32stack_at_exp23_recipe.sh`
Outdir: `out/exp33_aug32stack_noenc_attn163264_bf16_mc88_256px_20k`

Running on Colab (~20 min wall-clock for 20k steps at 256px bs=4 on the
provisioned GPU — single iteration loop is fast enough that from-scratch is
the default for the follow-up experiments).

Results: TBD.

**Aug-scale risk**: exp24 crashed at `scale=4.0` (1/16 image area per crop →
hard-region recovery dip at step ~4k). exp33 uses U[1.0, 2.5] which is in
the regime that exp24 fell over in; watch the val curve at 4-6k steps for a
similar pattern. If present, follow-up with `--aug-scale-max 1.5`.

---

## exp34 — exp33 recipe + symmetric decoder spatial self-attention

**Status: WIRED 2026-05-14** (decoder-attn on the full-aug recipe; pair with
exp37 for the same architecture change on minimal aug)

Clean A/B vs exp33 (same data, same aug, same compute budget) to isolate
the contribution of putting `BottleneckAttention` on the decoder side at
the same resolutions as the encoder. Closes the long-standing asymmetry
in nanoWarp's UNet, which had spatial self-attn only in encoder + bottleneck
and never in the decoder (vs SD/SDXL convention which puts it symmetrically).

**Architecture delta vs exp33** — with `attn_resolutions = (16, 32, 64)`
and `image_size = 256`:
- `attn_dec4` at H/8 (32px), channels c3=352 — mirrors `attn4`
- `attn_dec3` at H/4 (64px), channels c3=352 — mirrors `attn3`
- `attn_dec2` at H/2 (128px) — None (128 ∉ attn_set)
- `attn_dec1` at H (256px) — None (256 ∉ attn_set)
- 16px bottleneck `mid_attn` already always on, unchanged.

Each `attn_dec*` is applied after the corresponding `dec*` ResBlock,
before any FiLM/tattn hooks.

**Gating**: `--use-decoder-attn` flag (default off → exp33 behaviour). When
on, the decoder attn modules are instantiated at exactly the same
resolutions as the encoder attn — single `attn_resolutions` knob controls
both.

**Recipe**: identical to exp33 except `--use-decoder-attn` is passed. Reads
the architectural delta on top of the full corruption-robustness aug stack;
exp37 below runs the same architecture change on the minimal-aug
(exp23-like) recipe so the delta can be read without aug confound.

```bash
WANDB_API_KEY=... bash scripts/run_exp34_decoder_attn_at_exp33_recipe.sh
```

Script: `scripts/run_exp34_decoder_attn_at_exp33_recipe.sh`
Outdir: `out/exp34_decoder_attn_noenc_attn163264_bf16_mc88_256px_20k`

Results: TBD.

---

## exp37 — exp23-equivalent recipe (minimal aug) + symmetric decoder attn

**Status: WIRED 2026-05-14** (architecture A/B on a clean baseline + a third
anchor for the corruption-Δ metric)

Two goals in one run:
1. **Architecture-only A/B against exp23 (lpips_vgg=0.234)**. exp34 stacks
   decoder attn on the full exp32 aug stack — which exp33's result showed
   costs ~0.07 lpips_vgg on clean before any architecture changes — so the
   architecture delta is hard to read. exp37 stacks the same architecture
   change on the proven low-aug exp23 baseline so the read is clean.
2. **Corruption-Δ anchor for a clean-trained 20k checkpoint.** The original
   exp25 checkpoint that would have filled this anchor was lost before the
   Δ metric was wired. exp37 provides a fresh clean-trained 20k anchor to
   compare against exp32's Δ=+0.064 (corruption-trained anchor).

**Aug settings ≈ exp23** (exp32 script reproduces exp23-style behaviour when
geometric/color aug is dialled to ~identity and corruption is fully skipped):
- `--aug-scale-min 1.0 --aug-scale-max 1.2` (matches exp23's
  `resize_scale=1.10` + `scale_jitter=0.10`)
- `--aug-rotate-deg 0.0`, `--aug-perspective-prob 0.0`
- `--aug-brightness 0 --aug-contrast 0 --aug-saturation 0` (no color jitter)
- `--clean-prob 1.0` → degradation pipeline fully skipped
- hflip stays at p=0.5 (matches exp23, no CLI knob)

**Architecture delta**: identical to exp34 (`--use-decoder-attn` → adds
`attn_dec3` at H/4=64 and `attn_dec4` at H/8=32, mirroring encoder attn3/4
at the same resolutions).

```bash
WANDB_API_KEY=... bash scripts/run_exp37_decoder_attn_at_exp23_recipe.sh
```

Script: `scripts/run_exp37_decoder_attn_at_exp23_recipe.sh`
Outdir: `out/exp37_decoder_attn_at_exp23_recipe_noenc_attn163264_bf16_mc88_256px_20k`

**What to look for** in `out/val_exp37_final_256px/val_metrics.json`:
- `mean_lpips_vgg_sampled` vs exp23's **0.234** — the decoder-attn delta on clean.
  - < 0.234 → decoder attn helps; stack on future runs.
  - ≈ 0.234 → neutral at this dataset size; capacity isn't the bottleneck.
  - > 0.234 → hurts; not enough data to train the extra modules.
- `delta_lpips_vgg` (corruption-val gap) — third anchor on the Δ axis next to
  exp25 (Δ=+0.116, clean-trained, 20k) and exp32 (Δ=+0.064, corruption-trained,
  20k). Same baseline aug as exp25 plus the architecture change.

Results: TBD.

**Independent of exp37, exp33c is wired** as the robustness-recipe test
(see entry below). exp33c sweeps the aug-recipe axis; exp37 sweeps the
architecture axis. Both can run in either order.

---

## exp33c — exp33b recipe (scale 1.5) with corruption tail dialled to a realistic web-video envelope

**Status: WIRED 2026-05-14**

Hypothesis: exp33b's clean-val regression vs exp23 (lpips_vgg 0.274 vs 0.234)
is partly the rare extreme tail of the corruption distribution burning
training capacity on inputs that never occur in real footage. Tightening the
tail should recover clean-val toward exp23 while keeping most of the
robustness gain (smaller corruption-val Δ than exp25's +0.116).

**Aug recipe deltas vs exp33b**:
- `--degrade-resize-min 0.25 → 0.5` (when fired: 4× area max instead of 16×)
- `--degrade-resize-max 0.75 → 0.9` (most resize cases very mild)
- `--corrupt-blur-max 3.0 → 2.0` (cuts out-of-focus extreme)
- `--corrupt-jpeg-min 30 → 40` (skips the worst block artifacts)

Everything else identical to exp33b: scale ∈ [1.0, 1.5], rotate ±25°,
perspective 0.15, color jitter ±0.3, hflip, exp23-style architecture (no
decoder attn / pyramid / DiT changes).

```bash
WANDB_API_KEY=... bash scripts/run_exp33c_milder_corruption.sh
```

Script: `scripts/run_exp33c_milder_corruption.sh`
Outdir: `out/exp33c_milder_corruption_noenc_attn163264_bf16_mc88_256px_20k`

**Goal**: clean lpips_vgg closer to exp23's 0.234 than exp33b's 0.274, and
corruption-val Δ still meaningfully below exp25's 0.116. Sweet spot is
clean ≈ 0.25 with Δ ≈ 0.08 — buys most of the robustness for half the
clean-val cost.

Results: TBD.

---

## exp35 — exp34 (or exp33) recipe + in-model source feature pyramid + FiLM

**Status: WIRED 2026-05-14** (ready to launch after exp34 settles)

Clean A/B vs the previous best of {exp33, exp34} (whichever wins) to
isolate the contribution of a permanent in-model source feature pyramid.
Single architectural delta on top of that baseline: adds `SourcePyramid`
+ per-decoder-level `FiLM` modulation.

**Architecture**:
- `SourcePyramid` ([src/img2img/source_pyramid.py](../src/img2img/source_pyramid.py)):
  4-stage conv pyramid (`stem → s1 → s2 → s3`) run **once per source per
  forward pass**, producing features at the four decoder resolutions
  (H, H/2, H/4, H/8) with channels matching UNet widths
  (c1=88, c2=176, c3=352, c4=352). ~1.8M params at mc=88.
- `FiLM` (same file): per-decoder-level 1×1 conv produces (γ, β) from a
  pyramid feature; decoder activation becomes `x * (1 + γ) + β`. Both γ and
  β zero-init → identity at init. ~573k params total across the 4 levels.
- Net: ~2.4M extra params over the baseline (49M → ~51M).

**Why FiLM not cross-attention** (first try): half the params, no head-dim
hyperparams, identity-at-init for free. If FiLM shows even modest gain we
can iterate to cross-attn later.

**Why no inference-time external deps**: pyramid is part of the UNet and
ships in the same checkpoint as everything else. Permanent in-model
alternative to the optional ResNet18 source encoder.

**Recipe**: same as the baseline plus `--use-source-pyramid`. If exp34
wins, also pass `--use-decoder-attn` to stack on top.

```bash
WANDB_API_KEY=... bash scripts/run_exp35_pyramid_at_exp33_recipe.sh
```

Script: `scripts/run_exp35_pyramid_at_exp33_recipe.sh`
Outdir: `out/exp35_pyramid_film_noenc_attn163264_bf16_mc88_256px_20k`

**Compute redundancy**: pyramid output is independent of t, so it
recomputes wastefully across the 20 ODE steps at inference. Pyramid is
~1.8M cheap conv params → <1ms per call on a 4090 → ~600ms wasted per
30-frame video clip. Acceptable for now; can cache later if profiling
demands it.

Results: TBD.

**Architecture diagram**: [docs/model_architecture.html](model_architecture.html)
(self-contained HTML/SVG; shows the full UNet, the sub-block structure
within each level — ResBlock → attn? → FiLM? → tattn? — and the exp35
source pyramid + FiLM hooks).

---

## exp36 — exp33 recipe + DiT bottleneck

**Status: WIRED 2026-05-14** (ready to launch after exp33/34/35 land)

Replaces the convolutional bottleneck (`mid_attn` + `mid2 ResBlock`) with a
stack of 4 DiT-XL-style transformer blocks operating on the flattened
(H/16 × W/16, cm=704) token grid. `mid1` (the c4 → cm channel-widener
ResBlock) is preserved upstream so the DiT stack always sees constant width.

**Block structure** (per DiT block):
- `LayerNorm` (elementwise_affine=False) → adaLN-zero modulation
  (shift_msa, scale_msa, gate_msa from a single Linear(t_emb_dim → 6·D)
  with zero-init weight and bias)
- MHSA (qkv proj + scaled_dot_product_attention + out proj)
- gated residual
- `LayerNorm` → adaLN-zero modulation (shift_mlp, scale_mlp, gate_mlp)
- MLP (Linear → GELU → Linear, hidden=4D)
- gated residual

Zero-init adaLN gates → block emits its input unchanged at step 0 →
no-DiT checkpoints load cleanly via strict=False, identical at-init forward.

**Positional embeddings**: 2D sinusoidal, size-agnostic so the same DiT
stack works at the 8×8 / 16×16 / 32×32 bottleneck grids that arise from
the 128 / 256 / 512px training phases.

**Heads**: auto-picked to keep `head_dim` a Flash-attention-friendly
power of 2. At cm=704: head_dim=64, num_heads=11.

**Param cost**: ~28M added (49M → ~77M total). Outside "same param budget"
territory — explicit choice to test maximum DiT capacity. Halve to ~14M
with `--num-dit-blocks 2` if exp36 is competitive on quality but the param
bloat is a problem for downstream temporal fine-tuning.

**Recipe**: identical to exp33 except `--use-dit-bottleneck` (+ optional
`--num-dit-blocks` and `--dit-mlp-ratio`).

```bash
WANDB_API_KEY=... bash scripts/run_exp36_dit_bottleneck_at_exp33_recipe.sh
```

Script: `scripts/run_exp36_dit_bottleneck_at_exp33_recipe.sh`
Outdir: `out/exp36_dit_bottleneck_noenc_attn163264_bf16_mc88_256px_20k`

Results: TBD.

**Why DiT over windowed attention at higher resolutions**: the windowed-attn
path (see [model_architecture.html](model_architecture.html) notes) would
add spatial-attention coverage at 128/256px levels that aren't currently
attended; DiT instead changes the *kind* of mixing at the bottleneck where
attention is already happening. The captain's log discussion concluded that
SD/SDXL conventionally skip full-res attention because the source-in-stem
path already provides full-res spatial information, so adding higher-res
attention is mostly buying high-frequency texture coherence rather than new
spatial reasoning. DiT at the bottleneck is the "smarter mixing at the
right resolution" direction instead.

---

## exp37 — exp23-equivalent recipe (minimal aug) + symmetric decoder attn

**Status: WIRED 2026-05-14**

Two goals in one run:

1. **Architecture A/B on a clean baseline.** exp34 stacks symmetric decoder
   attn on top of exp33's full aug stack (which itself regressed clean-val
   from 0.234 → 0.308 lpips_vgg). exp37 puts the same architecture change
   on top of an exp23-style minimal-aug recipe, isolating the attn delta
   from the aug delta. If exp37 < exp23 (0.234) on clean lpips_vgg, the
   symmetric attn is a real win regardless of aug recipe.
2. **Clean-trained Δ-reference at 20k steps.** The lost exp25 step-20k
   checkpoint motivated this — exp37 gives a fresh clean-trained 20k
   snapshot with the new Δ metric measured from the very first val pass,
   establishing where the "no corruption training" floor sits on the Δ
   axis (expected ∼0.10–0.12 lpips_vgg, matching the prior exp25 result).

**Aug settings** (reproducing exp23-style behaviour inside the
`train_exp32_prog512.py` script):
- `scale ∈ [1.0, 1.2]` (≈ exp23's `resize_scale=1.10` + jitter 0.10)
- `rotate ±0°`, `perspective_prob=0`, all color jitter at 0
- `clean_prob=1.0` → degradation pipeline fully skipped
- `hflip_prob=0.5` (hardcoded default, matches exp23)

**Architecture delta** vs exp23: `--use-decoder-attn` adds `attn_dec3`
and `attn_dec4` (the two encoder-attn levels mirrored on the decoder
side). +~3M params over the exp23 backbone.

```bash
WANDB_API_KEY=... bash scripts/run_exp37_decoder_attn_at_exp23_recipe.sh
```

Script: `scripts/run_exp37_decoder_attn_at_exp23_recipe.sh`
Outdir: `out/exp37_decoder_attn_at_exp23_recipe_noenc_attn163264_bf16_mc88_256px_20k`

Results: TBD.

**Reading the result**:
- exp37 clean lpips_vgg < 0.234 → symmetric decoder attn helps even at
  minimal-aug; promote to default in future runs.
- exp37 ≈ 0.234 → attn change is a wash; the apparent improvement from
  decoder attn (when comparing exp34 vs exp33) would be aug-interaction,
  not architecture.
- exp37 > 0.234 → attn hurts at clean training; the encoder-only
  attention pattern was already enough.
- exp37 Δlpips_vgg ≈ 0.10–0.12 → clean-trained models always have a big
  robustness gap regardless of attn pattern (confirms Δ is an
  aug-recipe lever, not an architecture one).

---

## Temporal / video experiments

All temporal experiments (exp27 onwards) are documented in
[captains_log_video.md](captains_log_video.md).

---

## dual-LPIPS metric (2026-05-11)

validate.py now reports both `mean_lpips_squeeze_sampled` (continuity with
exp01–15) and `mean_lpips_vgg_sampled` (out-of-loop honest check). 28% gap
observed on exp08-noenc (0.163 sq vs 0.209 vgg). Always report both going forward.

---

## Corruption-robustness Δ-metric (2026-05-14)

Both the in-loop val ([train_exp32_prog512.py](../experiments/010_img2img_photo2comics/train_exp32_prog512.py))
and the final validation pass ([validate.py](../experiments/010_img2img_photo2comics/validate.py))
now run a second sampling pass per val batch against a deterministically
corrupted source, alongside the standard clean-source pass.

**Corruption** ([src/img2img/metrics.py::val_corrupt](../src/img2img/metrics.py)):
fixed mid-strength so numbers are comparable across runs/steps —
`resize 0.5× then bilinear-up → JPEG q=60 roundtrip → Gaussian blur σ=1.0`.
Roughly equivalent to a moderately-compressed web image. Strictly easier
than the worst-case training corruption (which goes up to σ=3 / q=30 /
resize 0.25× in exp32-style training).

**New metrics in `val_metrics.json` and wandb**:
- `mean_lpips_{squeeze,vgg}_corrupted` — same val data, corrupted source.
- `delta_lpips_{squeeze,vgg}` — corrupted minus clean. Smaller = more robust.

**Interpretation**: clean-trained models (exp23/exp25) show big Δ (∼0.10
lpips_vgg) because corruption is OOD for them. Corruption-trained models
(exp32+) show ∼0.05–0.07. The Δ distills "how much does this model fall
apart when the input isn't pristine" into one number, retroactively
backfillable by re-running validate.py on saved checkpoints.

**Reference numbers** (each model at its step 20k checkpoint, val at 256px,
EMA, 25 batches × bs=4, sample_steps=20):

| run | aug stack | clean lpips_vgg | corrupt lpips_vgg | **Δ lpips_vgg** | clean lpips_sq | corrupt lpips_sq | Δ lpips_sq |
|------|-----|------|------|------|------|------|------|
| exp25 (step 20k) | scale=1.10 + hflip | **0.234** | 0.350 | **+0.116** | 0.128 | 0.215 | +0.088 |
| exp32 (step 20k) | scale∈[1.0,2.5] + full corruption | 0.265 | **0.329** | **+0.064** | 0.142 | **0.185** | +0.043 |

Crossover: on corrupted source, exp32 wins (0.329 vs 0.350 lpips_vgg);
on clean source, exp25 wins (0.234 vs 0.265). The corruption-trained model
trades ~13% clean-quality for ~46% smaller robustness Δ and outright wins
when input isn't pristine — which is the regime real video frames live in.

**Going forward**: every `run_exp*.sh` end-of-run validate.py call now
emits both clean and corrupted numbers + Δ. The Δ should be reported
alongside lpips_vgg in any A/B summary; if Δ is large, the clean-val
number alone is misleading about real-world quality.

---

## Known bugs / lessons

- **2026-05-11 lesson: GAN weight 0.1 dominates over LPIPS 0.2.**
  At hinge equilibrium g_gan~1.5, contribution = 0.1×1.5 = 0.15 vs LPIPS
  contribution ~0.2×0.06 = 0.012. GAN is 12× larger than LPIPS in practice.
  For GAN as a weak regulariser, target gan_weight so that g_gan contribution
  ≈ LPIPS contribution, i.e. gan_weight ≈ 0.012/1.5 ≈ 0.008. 0.005 is in range.

- **2026-05-11 lesson: native-res crops (scale=4.0) cause 6k-step loss spike
  recovery.** 256px crops from 1024px images capture 1/16 image area — too
  sparse. Background-heavy crops dominate early training; face-heavy crops
  appear later and cause a loss jump. scale=2.0 (1/4 area) avoids the spike
  but still underperforms scale=1.10 (~full image). Confirmed ranking:
  scale=1.10 > scale=2.0 >> scale=4.0 on lpips_vgg (0.234 vs 0.304 vs 0.449).

- **2026-05-11 lesson: mc=176 hits capacity ceiling at 908 pairs/256px.**
  Both resize_conv and pixel_shuffle upsamplers showed periodic artifacts.
  Model converges to identity mapping. mc=88 (44M params) is the practical
  limit for this dataset scale.

## Open follow-ups

- ~~Compute "source-as-prediction" baseline SSIM/LPIPS.~~ **Done 2026-05-10.**
  Floor at 128px val: SSIM 0.617 / LPIPS 0.199. Floor at 256px val:
  SSIM 0.557 / LPIPS 0.299. Best 128px model: SSIM 0.740 / LPIPS 0.153.

- ~~Trainer `--resume` support.~~ **Done 2026-05-10.** Checkpoints
  (intermediate and final) include model + EMA + optimizer state + step.
  See exp08 resume command in its section.

- **Rename `FlowConfig.timesteps`** → `time_embedding_scale`. Currently
  `timesteps=1000` is misleading because FM has no discretized timestep
  schedule — `t` is continuous in [0,1] and we multiply by 999 only for
  the sinusoidal `TimeMLP`. Cleanup-only; touches checkpoint configs.

- **Best-checkpoint selection (val-LPIPS early stopping).** exp10/exp11
  showed LPIPS can regress past the optimum while SSIM/MSE keep improving.
  A trainer hook that re-validates every checkpoint and saves
  `best_lpips.pt` would automate finding the optimum. ~30 lines.

- Replace `save_loss_plot` with log-y + rolling mean. Lower priority now
  that wandb handles smoothing.

- Latent flow matching once the pixel-space pipeline is solid. Big lift
  (need an autoencoder), parked for when we have the bigger dataset
  results in hand.

- **Ablate `freeze=all` vs `freeze=partial`** on the encoder-on path now
  that we know freeze=all is stable. exp08-lpips uses freeze=all; checking
  whether freeze=partial would have worked too (and unlocked the deeper
  ResNet layers for task adaptation) would be informative.

- **Cheap SOTA-inspired patterns to consider** (post-exp14, see survey
  earlier in this document):
  - Zero-init gate on `FuseBlock` output (ControlNet trick): ~3 lines.
    Training stability.
  - Multi-scale L1/L2 loss alongside main loss (SR convention): ~10 lines.
    Small perceptual gain.
  - Charbonnier loss replacing pure MSE in the FM/diffusion path:
    trivial, minor numerical-stability benefit.
  - Window attention (SwinIR-style) at high resolutions: medium effort,
    relevant if we go ≥ 512px.

