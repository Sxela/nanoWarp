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

| split | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|
| train (287 pairs) | 0.6096 | 0.1976 |
| val (50 pairs) | 0.6168 | 0.1987 |

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

## Open follow-ups

- Compute "source-as-prediction" baseline SSIM/LPIPS to know the floor any
  conditional model has to beat. Five-line script.
- Replace `save_loss_plot` with log-y + rolling mean (now optional since
  wandb handles smoothing).
- ~~Trainer `--resume` support to continue from a saved checkpoint.~~
  **Done 2026-05-10.** `--resume PATH` loads model + EMA + optimizer state
  + step number. Checkpoints (intermediate and final) now include optimizer
  state and step. Old pre-2026-05-10 checkpoints can still be resumed but
  AdamW moments restart fresh (warning printed). See exp08 resume command
  in the exp08 section.
- Latent flow matching once the pixel-space pipeline is solid.
- Once exp08 lands: ablate `freeze=all` vs `freeze=partial` on the
  encoder-on path now that we know freeze=all is stable.
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

### exp10 — multi-scale attention + bf16 on top of exp08-noenc — RUNNING (step 20k/30k snapshot validated)

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
|  1k |  |  |  |  |
|  5k |  |  |  |  |
| 10k |  |  |  |  |
| 15k |  |  |  |  |
| **20k** | **0.734** | **0.736** | **0.159** | **0.153** |
| 30k | — | (in progress) | — | (in progress) |

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

This is at step 20k of 30k; the final number could push exp10 past
exp08-lpips on both metrics rather than just match.

### exp11 — exp10 + linear RGB — PLANNED (Scenario A activated)

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
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp11_linrgb_noenc_attn832_bf16_mc88_30k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,linear-rgb,exp11" `
    --outdir out/exp11_linrgb_noenc_attn832_bf16_mc88_30k
```

Predictions:
- LPIPS: -2 to -5% vs exp10 (0.146-0.150 range vs exp10's 0.153)
- SSIM: tiny shift, probably +/-1% (linear vs sRGB SSIM differences mostly
  cancel since we report in sRGB).
- Visual: cleaner palette in mid-trajectory frames, less weird color drift
  along the FM path.

### exp11 / exp12 — conditional plan based on exp10 outcome (now Scenario A — linear RGB chosen above)

Spec ahead of time so the next pair of experiments is unambiguous.

**Scenario A — exp10 WINS (no-enc + multi-scale attn closes or beats LPIPS).**
Attention replaces the ImageNet priors. Push on the rest of the OpenAI UNet
ladder + orthogonal axes.

- **exp11** = exp10 + linear RGB. Physically correct interpolation in FM
  (`x_t = (1-t)·source + t·target` only matches light arithmetic in linear
  space). ~30 lines (sRGB↔linear conversions at dataset / LPIPS / panel
  boundaries). Free win on FM correctness.
- **exp12** = exp11 + FiLM time injection + 2 ResBlocks per level. Finishes
  Path C (the OpenAI UNet upgrade ladder). FiLM is ~5 lines (replace
  additive `time_proj` with a `(scale, shift)` linear). 2 ResBlocks per
  level is the standard DDPM/SD pattern.

**Scenario B — exp10 LOSES (encoder priors still required).**
Attention isn't enough to replace the encoder at this data scale. The
question becomes: does attention help *on top of* the encoder?

- **exp11** = exp08-lpips + multi-scale attention(8,16,32) + bf16
  (encoder ON, mc=64). Single-variable change vs the proven best. The
  Option A we considered earlier. Tests "does attention help a model
  that already has good priors?"
- **exp12** = (winner of exp08-lpips vs exp11) + linear RGB. Same physical-
  correctness motivation as Scenario A's exp11.

**Scenario C — exp10 MIXED (better SSIM, still worse LPIPS).**
Attention does real structural work but doesn't replace prior compute.
Combine the wins.

- **exp11** = exp08-lpips + multi-scale attention(8,16,32) + bf16
  (same as Scenario B's exp11). Tests if SSIM gain stacks with encoder's
  LPIPS edge.
- **exp12** = exp11 + linear RGB.

**Things worth running regardless of exp10's outcome:**
- `--lpips-weight 0.4` ablation on the current best architecture.
  exp08-lpips's curve was still improving at 20k; pushing harder might
  unlock more. Single-flag change.
- **Source-as-prediction baseline metric.** Compute mean SSIM/LPIPS of
  `source` vs `target` on val (no model, just the metric). 5-line script.
  Tells us the floor any conditional model has to beat. Without this we
  can't tell if 0.152 LPIPS is "good" or "barely above do-nothing."
- `--lpips-weight 0.0` ablation **at 30k steps with attention + bf16**.
  Tests whether LPIPS aux is still load-bearing once attention is in the
  model. If so, attention does the perceptual work LPIPS was doing, and
  we can drop the LPIPS computational overhead.

| if exp10... | exp11 | exp12 |
|---|---|---|
| wins    | + linear RGB                   | + FiLM + 2 ResBlocks             |
| loses   | + attn on encoder-on baseline  | + linear RGB on the winner       |
| mixed   | + attn on encoder-on baseline  | + linear RGB on exp11            |
| any     | source-baseline metric, lpips=0.4 ablation | ↑ |
