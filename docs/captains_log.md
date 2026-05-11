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

### exp17 — exp15 minus LPIPS-squeeze (test the redundancy) — PLANNED

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

Predictions:
- If exp15 val LPIPS-squeeze < exp10's 0.153 by > 3% → Gram texture
  matching was a real bottleneck. Texture-aware loss is the way forward
  for stylization tasks.
- If exp15 ≈ exp10 → LPIPS already captured what Gram does for our
  task (unlikely but possible). Stay with simpler LPIPS aux.
- Visual check is more informative than the LPIPS number here — the
  expected gain is "stronger anime stylization" which may not score on
  pairwise LPIPS but is the qualitative win.

### exp16 — GAN aux loss for shape complexity — DRAFTED (not yet implemented)

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
- This spec exists so we know what "exp16" means if we need to escalate.

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
    --wandb --wandb-tags "flow,no-encoder,lpips,gan,exp16" `
    --outdir out/exp16_gan_noenc_attn832_bf16_mc88_30k
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

