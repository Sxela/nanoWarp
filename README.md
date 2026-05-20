# nanoWarp

A living journal for building **small, fast image-to-image and video-to-video models**.

Not a giant foundation-model playground. Not a paper zoo.
A focused repo for figuring out how far we can push **structure-preserving visual transformation** with compact models and visual-first tooling.

---

## Current status (2026-05-19)

**65+ experiments deep**, single canonical line on flow-matching img2img at ~50M params.

| canonical | recipe | val_portraits face_lpips_sq | Δ_lpips_vgg | use case |
|---|---|---|---|---|
| **exp65b** (80k, x0-pred) | minimal aug + cross-attn @ H/8 + x0-prediction | **0.0996** | 0.047 | **quality canonical** — best face quality + huge SSIM jump |
| **exp61** (80k) | mid aug + cross-attn @ H/8 (STACK) | 0.103 | **0.025** | deployment canonical — best real-world robustness |

**exp65b is the new quality canonical**: x0-prediction (predicting clean target directly instead of velocity) gave the biggest single-experiment quality win of the entire 3k era — wins on every quality metric including **+29% SSIM** and **-10% legacy val face_lpips_sq**. Ties exp60 on face_lpips_sq to 4 decimals. The tradeoff is robustness Δ (+17-21% regression). Mid-aug component from exp61 still owns the robustness lever.

**exp65c stack** (x0-pred + mid-aug + cross-attn at 80k) is the open question: if orthogonal composition holds, it would deliver best quality AND best robustness in one model — the single canonical.

The current architecture under both canonicals:

- **Rectified flow matching** (not diffusion — see exp54/55 for why diffusion structurally underperforms at this scale)
- **~50M-param UNet** at mc=88, source-in-stem, no source encoder
- **Multi-scale self-attention** at H/2, H/4, H/8 in both encoder and decoder (exp10 / exp34)
- **In-model SourcePyramid** (4-stage conv) + **FiLM** modulation at every decoder level (exp35)
- **Cross-attention** at H/8 decoder level pulls source info from pyramid features (exp59)
- **3k mixed dataset** (908 Flux-synth + 2321 FFHQ portraits) — real-photo sources matter
- **80k training steps @ 256px, bs=4**, bf16, LPIPS-VGG aux 0.2, EMA 0.999

See `journey/exp59/architecture.html` and `journey/exp34/architecture.html` etc. for inline SVG architecture diagrams emitted only at each arch change.

---

## Where to look

| | |
|---|---|
| [docs/captains_log.md](docs/captains_log.md) | Legacy 1k-synth era (exp01-exp49). Every experiment's recipe, motivation, hypothesis, result, lesson. |
| [docs/captains_log_3k.md](docs/captains_log_3k.md) | Current 3k-mixed era (exp50-exp62). Same format; this is the active log. |
| [docs/results_table.md](docs/results_table.md) | Cross-experiment fast-reference tables. Two splits (legacy val + val_portraits), two eras (1k synth + 3k mixed). |
| [journey/](journey/) | Per-experiment folders with descriptions, run scripts, val metrics, panels, and `architecture.html` (only at arch-change boundaries). |
| [CLAUDE.md](CLAUDE.md) | Binding working principles (reproducibility, exp isolation, smoke-test mandates, gotchas). |
| [scripts/run_exp*.sh](scripts/) | Every experiment has a reproducible bash script. One script = one set of CLI flags = one outdir. |

---

## Thesis (unchanged from project start)

The interesting problem is **not** random-noise-to-world generation.
It is structure-preserving translation:

- photo → comics / anime / stylized
- video → stylized video
- strong preservation of layout / pose / identity / timing
- small enough to train and iterate on quickly

**The core bet**: these tasks should allow much smaller trainable systems than text-to-image generation, because the input already gives us structure.

Validated through 62 experiments:
- 50M params is sufficient at this task (no need for SDXL/Flux scale)
- pure-pixel DiT is dead at this scale (exp47/48); conv UNet wins
- diffusion is structurally bottlenecked vs flow at small scale (exp54/55)
- data scale (1k → 3k pairs, real-photo sources) was the biggest single lever
- canonical post-exp52: extending training duration (20k → 80k) + cross-attn conditioning
- mid-aug exposure to corruption gives 40% robustness improvement essentially for free

---

## Design rules (binding)

1. **Reproducibility**: every training run has a `scripts/run_expNN_*.sh` with all flags pinned. No ad-hoc `python train.py` invocations.
2. **No secrets in committed files**: `WANDB_API_KEY` etc. come from the launching shell via `: "${WANDB_API_KEY:?...}"`.
3. **One exp number per recipe**. Variants get letter suffixes (exp33b, exp58b).
4. **Single-checkpoint inference**: no external pretrained backbones at inference time. LPIPS-VGG (training-only) and Flux (data-prep-only) are allowed; everything else has to live in the `.pt`.
5. **Smoke tests are mandatory** before declaring code-changes done. Catch errors in 1-2 minutes locally, not 20 minutes into a Colab run.
6. **Visual-first logging**: PNGs, panels, comparison grids, GIFs. wandb is logged to but isn't the source of truth.

Full versions of these and more in [CLAUDE.md](CLAUDE.md).

---

## Architecture (current canonical)

```
input: x_t = (1-t)·source + t·target  (flow interpolant)

         ┌──────────────────┐
source ──┤ SourcePyramid    │  4-stage conv, ~1.8M params
         │ (in-model)       │  produces features at H, H/2, H/4, H/8
         └────────┬─────────┘
                  │ pyramid feats
                  ▼
[noisy_target] + [source] ──┐   ┌──> in_conv (6→88) ─┐
                            │   │                    │
                            └───┘                    │
                                                     ▼
                                            ┌─── UNet Encoder ───┐
                                            │  4 down stages     │
                                            │  attn @ 16/32/64   │
                                            │  bottleneck @ 16   │
                                            └─────┬──────────────┘
                                                  │
                                            ┌─── UNet Decoder ───┐
                                            │  4 up stages       │
                                            │  attn_dec @ 16/32/64
                                            │  FiLM (per level)  │ ◄── pyramid features
                                            │  cross-attn @ H/8  │ ◄── pyramid f3 (KV)
                                            └─────┬──────────────┘
                                                  ▼
                                         out: velocity v(source, x_t, t)

x_{t+dt} = x_t + dt · v(...)   (Euler ODE, 20 inference steps)
```

Knobs that have been thoroughly explored (and the verdict):

| knob | exp | verdict |
|---|---|---|
| ResNet18 source encoder | exp01-07 | dropped in exp08; trainable scratch matches |
| pixel_shuffle upsample | exp09 | didn't beat resize_conv |
| multi-scale attn (16/32/64) | exp10 | clear win, retained |
| decoder self-attn (mirror encoder) | exp34 | retained as part of exp35 baseline |
| source pyramid + FiLM | exp35 | new canonical from here forward |
| DiT bottleneck | exp36 | marginal (+28M params for 1% improvement) |
| contrastive source loss | exp38/39 | wash |
| VGG Gram style loss | exp40 | regressed |
| CFG at inference | exp41 | cratered (flow != diffusion) |
| LPIPS-anneal schedule | exp42/45 | wash |
| σ_noise > 0.05 | exp43 | catastrophic at 0.30 |
| 128→256→512 progressive | exp46/49 | EMA contamination issue |
| pure-pixel DiT | exp47/48 | dead at this scale |
| LANCZOS resize | exp53 | wash |
| diffusion (eps) re-test | exp54/55 | structurally worse than flow at this scale |
| mid aug (head pose + mild corrupt) | exp56 | +40% robustness for ~3% clean-quality cost |
| source dropout (regularization) | exp57 | tie + small robustness win |
| logit-normal t-sampling | exp58/58b | doesn't transfer from text2img (endpoint starvation) |
| **cross-attn conditioning** | exp59/60 | clean uniform win at +500k params |
| **stack mid-aug + cross-attn** | exp61 | best robustness ever, ties best quality |
| drop source-in-stem + multi-scale xa | exp62 | small portrait quality win, +17% robustness regression (source-in-stem is a robustness feature) |
| PatchGAN adversarial on exp61 (gan_w=0.02 + adaptive switch) | exp63 | metrics moved within noise; visually subtle drift, NOT texture sharpening. gan_weight too weak + adaptive switch starved D. Retry pending. |
| AdaLN-Zero time conditioning everywhere | exp64 / exp64b | **LOSES at 80k** (+11% face_lpips_sq, +46% Δ vs exp60). Modern transformer-arch tricks don't auto-port to small conv UNets. Chapter closed. |
| **x0-prediction (vs velocity)** | exp65 / exp65b | **NEW QUALITY CANONICAL at 80k**: ties exp60 face_lpips_sq, +29% whole_ssim portraits, -10% legacy face_lpips_sq, +21% robustness Δ regression. Biggest single-experiment quality win of the 3k era. |
| wider mc=128 capacity test | exp66 / exp66b | **LOSES at 80k** (+5% face_lpips_sq, +26% Δ vs exp60). 50M is the right size for 3k data; more params overfits. |
| SGDR 2-cycle LR schedule | exp67 / exp67b | **LOSES at 80k** (+4% face_lpips_sq). Warm restart disrupts productive late-training refinement; the late-cosine "plateau" was actually useful. LR-schedule axis closed alongside LR-value axis. |

---

## Training (current recipe)

```bash
export WANDB_API_KEY=...
# Quality canonical (clean 80k benchmark winner):
bash scripts/run_exp60_cross_attn_at_exp52_recipe_80k.sh

# Deployment canonical (best real-world robustness):
bash scripts/run_exp61_cross_attn_plus_mid_aug_80k.sh
```

Each script:
- Auto-resumes from the latest checkpoint in its outdir (interrupt-safe)
- Streams to wandb under `nanoWarp` project
- Validates twice at end: `val` split (legacy continuity) + `val_portraits` (FFHQ portraits, the meaningful face signal)
- Saves checkpoints every 10k steps, "best" snapshots every 5k steps
- Takes ~80 minutes on a Colab T4 / A100

---

## Validation

```bash
PYTHONPATH=. python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint <path>/expNN_model.pt \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 \
    --sample-steps 20 --use-ema \
    --outdir out/val_$NAME_on_val_portraits
```

Outputs `val_metrics.json` (lpips_sq, lpips_vgg, ssim, face_lpips_sq, face_lpips_vgg, face_ssim, plus corrupted-source variants and their deltas) and val panels at the same image size.

For 3k-era runs, validate on both `val` (legacy continuity) and `val_portraits` (200 FFHQ portraits — the meaningful face signal). Cite both numbers.

---

## Setup

```bash
./install.sh
source .venv/bin/activate
```

Main dependencies in `requirements.txt`. Tested on Python 3.12 and 3.14 (Windows host + Colab Linux training).

---

## Repo layout

```text
nanoWarp/
  README.md                          # this file
  CLAUDE.md                          # binding working principles
  docs/
    captains_log.md                  # legacy 1k-synth era
    captains_log_3k.md               # active 3k-mixed era
    results_table.md                 # cross-experiment fast reference
  data/
    photo2anime_3k/                  # 3229 train + 100 val + 200 val_portraits pairs
    photo2anime/                     # legacy 1k-synth
  src/img2img/
    model.py                         # Img2ImgDiffusionUNet
    flow.py                          # RectifiedImageFlow
    diffusion.py                     # GaussianImageDiffusion (kept for exp54/55 path)
    source_pyramid.py                # SourcePyramid, FiLM, CrossAttnCond
    dit.py, dit_pixel.py             # DiT variants (exp36, exp47)
    metrics.py                       # ValidationMetrics, val_corrupt, face_crops
    ckpt.py                          # build_model_from_ckpt (auto-detects arch from state_dict)
  experiments/010_img2img_photo2comics/
    train_exp32_prog512.py           # current trainer (3k era)
    train.py                         # legacy trainer (1k era)
    validate.py                      # shared validation
    infer_video.py                   # video inference
  scripts/
    run_expNN_*.sh                   # one script per experiment
    build_journey.py                 # populates journey/ from artifacts
    build_journey_arch.py            # emits architecture.html at arch boundaries
    download_diverse_sources.py      # CelebA-HQ + Places365 pull for data scale-up
    merge_ffhq_into_photo2anime.py   # builds the 3k mixed dataset
  journey/
    expNN/
      description.md                 # captain's log section for this exp
      run_script.sh                  # hardlink to scripts/run_expNN_*.sh
      val_metrics_*.json             # sanitized val outputs
      panels/                        # gitignored — hardlinks rebuilt by build_journey.py
      architecture.html              # ONLY at arch-change boundaries
```

---

## Notable principles encoded over 60+ experiments

- **In-model architecture, no inference-time external deps**: the entire pipeline ships in the `.pt`.
- **`x_t = (1-t)·source + t·target` matters**: source is already implicit in the flow interpolant, which is why source-in-stem may be redundant (exp62 tests this).
- **Stateful torchmetrics LPIPS needs `.reset()` before each call**: silent throughput collapse if forgotten.
- **Windows console is cp1252**: no unicode arrows or em-dashes in print statements that target stdout.
- **EMA decay 0.999 + progressive training don't mix**: EMA gets polluted across resolution phases.
- **Val resolution must match training resolution** for honest numbers; misaligned val is misleading.
- **CFG doesn't transfer from diffusion to flow** (in flow, v is a true velocity, can't be amplified).
- **Skin-tone bias in the Flux-generated targets** affects val_portraits comparison interpretation — small deltas (1-5%) are robust, medium deltas (10%) are partially confounded, large deltas (>30%) are too big to be pure bias.
- **50M params + simple additive time_proj is the recipe ceiling at 3k mixed data**: confirmed by *two* clean negative results — exp66b (mc=128 at 80k LOSES) and exp64b (AdaLN-Zero at 80k LOSES). Any +params architectural addition overfits at our data scale. To unlock more capacity, scale data first (10k+ pairs).
- **Modern transformer-arch tricks don't auto-port to small conv UNets**: DiT/SD3-style AdaLN-Zero requires both transformer-heavy architecture AND much larger scale (DiT-XL 675M, SD3 8B) to pay off.
- **lr=2e-4 is well-calibrated for the canonical recipe** (exp68a confirmed): even 2× (lr=4e-4) caused +29% face_lpips_sq regression. The LR axis is closed; the default that came down from exp01 era happens to be near-optimal.

---

## What's next

Active follow-ups documented in [docs/captains_log_3k.md](docs/captains_log_3k.md) under "Open follow-ups":

- More diverse real-photo sources (CelebA-HQ + Places365, see [scripts/download_diverse_sources.py](scripts/download_diverse_sources.py))
- Resolution scale-up (currently 256px; sources are 512+)
- Skin-tone-stratified val split (to disambiguate model quality from dataset bias)

Stage 3 (V2V / temporal extension) hasn't begun in earnest yet — single-frame ceiling has been the priority.
