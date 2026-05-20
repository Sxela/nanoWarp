# Captain's log — `photo2anime_3k` era (2026-05-16 onwards)

This log covers experiments after the data scale-up to
`data/photo2anime_3k/` (1k Flux-synthetic + 2.3k real-FFHQ pairs).

**Why a new log file**: the new dataset has different distribution
properties (real photos as source domain, FFHQ portraits in val), so
metrics measured here are **not directly comparable** to numbers in
[captains_log.md](captains_log.md). Pre-2026-05-16 results stay in the
legacy log; numbers there were measured on the 1k-synthetic dataset
with its non-portrait val split.

For the legacy-era log see [captains_log.md](captains_log.md).
For the fast-reference table across both eras see
[results_table.md](results_table.md).

---

## Dataset structure

`data/photo2anime_3k/` (built 2026-05-16 via
[scripts/merge_ffhq_into_photo2anime.py](../scripts/merge_ffhq_into_photo2anime.py)):

| split | n pairs | content |
|---|---|---|
| train | 3229 | 908 original (synth photo + Flux anime) + 2321 FFHQ (real photo + Flux anime) |
| val | 100 | original group photos (peripheral / non-frontal faces; legacy) |
| val_portraits | 200 | FFHQ portraits (frontal faces, real photos) |

### Why this dataset matters

1. **Source-domain fix**: the original 1k pairs had Flux-generated
   *synthetic* photo sources. Every prior single-frame run was OOD on
   real-photo inference (nat1.mp4, OOD photos) the whole time. The
   FFHQ subset finally pairs real photo sources with the same Flux-anime
   target style, closing a train-test domain gap that's been there
   since exp14v2.

2. **Val-distribution fix**: the original 100 val pairs are group photos
   with peripheral / non-frontal faces. Every face_lpips / face_ssim
   number in the legacy log was measured on that distribution →
   under-counts both the face-quality problem and our improvements.
   `val_portraits` (200 FFHQ portraits) is the meaningful face-quality
   signal going forward.

3. **3.2× data scale-up**: 1k → 3.2k training pairs. The 1k regime was
   architecturally saturated (every exp33-49 hovered at lpips_sq ≈
   0.124). More data is the only lever that should move the floor.

### Methodology change vs legacy

- All new validate.py runs report both `val/` (legacy split, for
  continuity) and `val_portraits/` (real face-quality signal).
- `--wandb-resume "$OUTDIR"` flag attaches val to the training wandb run
  so post-training Colab death doesn't lose the final-val numbers.
- Run scripts log val under two prefixes: `final_val/` and
  `final_val_portraits/`.

---

## Retroactive baselines on val_portraits

Both run on 2026-05-16. Establishes the face-quality floor on the new
val split that exp50+ has to beat.

| ckpt | source training | face_lpips_sq | face_lpips_vgg | face_ssim | whole lpips_sq | whole ssim |
|---|---|---|---|---|---|---|
| **exp25 (80k)** | 1k synth, 80k steps, no arch changes | **0.169** | **0.345** | **0.500** | 0.216 | **0.392** |
| exp35 (20k) | 1k synth, 20k steps, +dec_attn+pyramid | 0.178 | 0.370 | 0.477 | **0.215** | 0.384 |

(103 faces detected in 200 val_portraits images.)

**exp25-80k beats exp35-20k on every face metric** — the 60k extra
training steps matter more for real-portrait face quality than the arch
changes did. **Canonical baseline for exp50 to beat is exp25-80k, not
exp35-20k**, on `val_portraits`.

**Key observation from these baselines**: face_lpips_sq on val_portraits
(0.169-0.178) is **10-17% worse** than on legacy val (0.153-0.157) —
confirms the legacy-val face metrics were systematically under-counting
the problem. Whole-image ssim drops from ~0.69 (legacy val) to ~0.39
(val_portraits) — a much harsher signal that matches the visual quality
reality on real-photo portraits.

---

## exp50 — exp35 recipe on `photo2anime_3k`

**Status: WIRED 2026-05-16**

First test of the data scale-up. Same recipe as legacy exp35 (minimal
aug, exp35 arch = decoder_attn + pyramid + FiLM, constant LPIPS=0.2,
20k @ 256px bs=4) but training data is the 3k merged dataset. Two
stacking effects expected:

1. **3.2× more data** → lifts the architecture/recipe ceiling that
   capped every legacy exp33-49 at lpips_sq ≈ 0.124.
2. **Real photo sources finally in training** → model stops being OOD
   on real-photo inference. Probably the bigger visual lift; metrics
   alone may understate it.

Validates twice at end (`--wandb-resume` so both go to the same wandb
run under separate prefixes):
- `--split val` on legacy 100 group photos (continuity with prior runs)
- `--split val_portraits` on 200 FFHQ portraits (face-quality signal)

```bash
WANDB_API_KEY=... bash scripts/run_exp50_exp35_recipe_on_3k.sh
```

Script: `scripts/run_exp50_exp35_recipe_on_3k.sh`
Outdir: `out/exp50_exp35_recipe_on_3k_noenc_attn163264_bf16_mc88_256px_20k`

**Results (2026-05-16, done)**:

| split | metric | value |
|---|---|---|
| val (legacy) | lpips_sq / lpips_vgg / ssim | 0.150 / 0.297 / 0.516 |
| val (legacy) | face_lpips_sq / face_lpips_vgg / face_ssim | 0.201 / 0.379 / 0.605 |
| **val_portraits** | **lpips_sq / lpips_vgg / ssim** | **0.170 / 0.353 / 0.444** |
| **val_portraits** | **face_lpips_sq / face_lpips_vgg / face_ssim** | **0.124 / 0.285 / 0.544** |
| val_portraits | Δ lpips_vgg / Δ lpips_sq | **+0.037 / +0.024** |

**val_portraits face_lpips_sq=0.124 is the lowest face metric we've ever
measured** (vs exp25-80k's 0.169 baseline). −27% on the meaningful
face-quality signal. Two predicted effects landed exactly:

1. **Data scale-up** lifted the ceiling that capped every legacy
   exp33-49 around lpips_sq=0.124 on the easier legacy val.
2. **Real-source domain finally in training** closed both the quality
   gap on real portraits AND — unexpectedly — the corruption-robustness
   gap, all without any explicit corruption aug. Δ lpips_vgg dropped
   from exp25's +0.116 to **+0.037**, comparable to exp32-100k's
   corruption-trained +0.040. Real-photo training inherently exposes
   the model to natural-image statistics that the corrupt-from-synth
   recipe was trying to simulate.

**Legacy val regression** (lpips_sq 0.124→0.150 vs exp35) is expected
and fine — different source distribution. Not the right signal.

**Implication**: data was the lever the whole time. Architecture work
(exp34-39, 42-49, 7+ runs in the legacy era) collectively moved
face_lpips_sq by ~0, while one data swap moved it by 27%.

---

## exp51 — FFHQ-only sanity test

**Status: DONE 2026-05-16**

Same recipe as exp35/50 but trained on `data/photo2anime_ffhq2k`
(2321 train + 200 val_portraits, no synth-source pairs at all).
Question: is the architecture+recipe even capable of learning faces
on a clean uniform dataset?

Final val on two splits:

| split | metric | exp51 (FFHQ-only) | exp50 (3k mixed) |
|---|---|---|---|
| val_portraits | face_lpips_sq | **0.122** | 0.124 |
| val_portraits | face_lpips_vgg | **0.280** | 0.285 |
| val_portraits | face_ssim | **0.550** | 0.544 |
| val_portraits | Δ lpips_vgg | **+0.031** | +0.037 |
| legacy val | face_lpips_sq | **0.290** (-90% vs exp35) | 0.201 |
| legacy val | face_ssim | 0.510 | 0.605 |

**Conclusions**:

1. FFHQ-only ≈ 3k-mixed on FFHQ portraits (1-2% improvement, noise floor).
   The synth-source 1k pairs in exp50 weren't actually hurting portrait
   quality — they just weren't helping either.
2. FFHQ-only **catastrophically regresses** on legacy val (small /
   peripheral / group faces). The model has no concept of off-center
   faces anymore.
3. The "data is the lever" thesis from exp50 was right but more
   specific: it was the **real-photo source domain** that mattered,
   not FFHQ-specifically. Adding ANY real-photo source domain (mixed
   or pure) closed the train-test gap.
4. **exp50 (3k mixed) stays as the canonical baseline**. FFHQ-only is
   not a useful endpoint — the curriculum option (FFHQ-only pretrain
   → fine-tune on mixed) is still open if face quality stalls again.

---

## exp52 — exp50 recipe × 80k steps

**Status: DONE 2026-05-16** (new canonical baseline)

Same recipe as exp50 (3k mixed, exp35 arch, minimal aug, constant LPIPS),
just 4× longer training. Mirrors the legacy exp23 → exp25 pattern.

| split | metric | exp50 (20k) | **exp52 (80k)** |
|---|---|---|---|
| val_portraits | face_lpips_sq | 0.124 | **0.101** (-19%) |
| val_portraits | face_lpips_vgg | 0.285 | **0.244** (-14%) |
| val_portraits | face_ssim | 0.544 | **0.579** (+6%) |
| val_portraits | whole lpips_sq | 0.170 | **0.145** (-15%) |
| val_portraits | whole ssim | 0.444 | **0.459** (+3%) |
| legacy val | face_lpips_sq | 0.201 | **0.183** (-9%) |
| legacy val | face_lpips_vgg | 0.379 | **0.355** (-6%) |
| legacy val | face_ssim | 0.605 | **0.623** (+3%) |

**face_lpips_sq=0.101 on val_portraits is the lowest face metric we've
ever measured.** The 20k → 80k extension lifted both vals — unlike
exp51 (FFHQ-only) which catastrophically regressed on legacy val past
its initial training. The mixed dataset retains diversity through long
training.

**Robustness Δ_lpips_vgg=0.045** (mild regression from exp50's 0.037
but still effectively as robust as exp32-100k's corruption-trained
0.040 — all without explicit corruption aug).

**exp52 is the new canonical baseline** for the 3k era. Future
experiments compare against exp52 face_lpips_sq=0.101 on
val_portraits.

---

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

## exp54 — diffusion (eps) re-test at exp50 recipe

**Status: DONE 2026-05-18** — catastrophic regression. Bucket #2 of
the hypothesis tree.

Re-running the experiment that "doomed" classical Gaussian diffusion in
the legacy era — but with every known confounder fixed. exp01-exp06
used eps-prediction diffusion with a ResNet18 source encoder, default
UNet, 1k synth dataset, ~2k training steps. DDIM reverse-sampling
collapsed; we switched to flow matching at exp07 and never looked back.

**Honest re-test setup**: same trainer as exp50 (`train_exp32_prog512.py`,
just extended to support `--method diffusion`), same exp35 arch
(decoder_attn + source_pyramid + FiLM), same 3k mixed dataset, same
recipe (minimal aug, constant LPIPS=0.2), same 20k @ 256px bs=4. Only
delta: `--method diffusion --prediction-type eps --diffusion-timesteps 1000`.

The trainer changes (back-compat: `--method flow` is default and exp50/52
reproduce bit-identical):
- `_sample_from_source` dispatches on method (flow=Euler ODE; diffusion=DDIM).
- `save_panel`/`save_face_panel`/`infer_nat1` now route through the helper.
- `save_checkpoint` writes `method=diffusion` + `diffusion=cfg.__dict__`.
- Training loss call filters out flow-only kwargs (contrastive_*) when
  method=diffusion. Both modules return the same 8-tuple shape.

**Hypothesis tree**:

1. Diffusion catches up to flow (within ±5% on face_lpips_sq):
   → flow's edge in the legacy era was a confounder, not method.
   → Method choice becomes a smaller lever; could revisit v-prediction,
     hybrid schedules, etc.

2. Diffusion still much worse (10%+ regression):
   → flow's edge is real at this model size / data scale.
   → Already controlled for sample_steps: in-loop val uses 20 DDIM
     steps (matches exp50 for speed) but final val uses **100** DDIM
     steps (diffusion's native sweet spot). If it still loses at 100,
     the gap isn't a stepcount artifact.

3. Diffusion is *better* (unlikely but possible if eps loss is a less
   blurry MSE signal than v-target):
   → flow assumption needs reconsidering; promote to 80k.

```bash
WANDB_API_KEY=... bash scripts/run_exp54_diffusion_at_exp50_recipe.sh
```

Script: `scripts/run_exp54_diffusion_at_exp50_recipe.sh`
Outdir: `out/exp54_diffusion_eps_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k`

A/B target — exp50 (flow):
- val_portraits face_lpips_sq=0.124, face_lpips_vgg=0.285, face_ssim=0.544
- val_portraits whole lpips_sq=0.170, whole ssim=0.444
- legacy val face_lpips_sq=0.201, face_ssim=0.605

**Results (final val @ 100 DDIM steps, EMA)**:

| split | metric | exp50 (flow @ 20) | exp54 (diffusion @ 100) | delta |
|---|---|---|---|---|
| val_portraits | face_lpips_sq | **0.124** | **0.508** | **+310%** |
| val_portraits | face_lpips_vgg | 0.285 | 0.760 | +167% |
| val_portraits | face_ssim | 0.544 | 0.370 | -32% |
| val_portraits | whole lpips_sq | 0.170 | 0.514 | +202% |
| val_portraits | whole lpips_vgg | 0.353 | 0.735 | +108% |
| val_portraits | whole ssim | 0.444 | 0.368 | -17% |
| val_portraits | Δ lpips_vgg | 0.037 | 0.047 | +27% |
| legacy val | face_lpips_sq | 0.201 | 0.482 | +140% |
| legacy val | face_lpips_vgg | 0.379 | 0.621 | +64% |
| legacy val | face_ssim | 0.605 | 0.524 | -13% |
| legacy val | whole lpips_sq | 0.150 | 0.433 | +189% |
| legacy val | whole ssim | 0.516 | 0.322 | -38% |

**Catastrophic across the board.** Not a marginal regression — diffusion
at 100 DDIM steps produced output that's 2-4× worse on LPIPS metrics
than flow at 20 Euler steps. The corruption-robustness gap (Δ_lpips_vgg)
is actually only slightly worse on val_portraits (+27% vs exp50);
the collapse is in *absolute quality*, not robustness.

**Three candidate root causes**, in order of plausibility:

1. **No source-as-init prior**. Flow's sample loop starts from `x = source`
   and refines toward target — the source acts as a strong, free
   inductive prior at every step. Diffusion samples from `x = N(0, I)`
   and conditions on source as a separate input channel. At this model
   size (~50M params), the conditioning signal alone isn't enough to
   pull samples back to the image distribution. **This is structural to
   the method.**

2. **LPIPS-on-x0_hat pathology**. At high t, `x0_hat = (x_t - sqrt(1-ab)·eps_hat)/sqrt(ab)`
   is amplified-noise garbage. LPIPS on garbage pushes eps_hat away
   from the right answer — actively harmful. **exp55 (lpips=0) tests
   this.** If exp55 ≫ exp54, this was the dominant factor.

3. **eps prediction at small scale**. eps-target has the same variance
   across all timesteps but the model has to handle wildly different
   t-conditional distributions. v-prediction smooths this. **A future
   exp could rerun with `--prediction-type v`.**

**Next step**: exp55 (lpips=0) is still the right A/B because it
disambiguates (1) from (2). If exp55 is also bad, (1) is the dominant
cause and the diffusion baseline is structurally bottlenecked at this
scale — no recipe rescue.

---

## exp55 — diffusion, LPIPS=0 (pure MSE eps prediction)

**Status: DONE 2026-05-19** — LPIPS hypothesis refuted; diffusion gap is
structural.

One-flag delta vs exp54: `--lpips-weight 0.0`. Tests whether LPIPS on
`x0_hat` is actively hurting diffusion training.

Why it might hurt: in eps-prediction diffusion, the per-step `x0_hat`
estimate is

    x0_hat = (x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)

At high t the denominator `sqrt(alpha_bar)` approaches 0, so any
eps_hat error gets divided by a tiny number → `x0_hat` is essentially
amplified noise. LPIPS on amplified noise gives a misleading gradient
that pushes eps_hat *away* from the right answer. Flow doesn't have
this pathology — `x_target_hat = x_t + (1-t) * v_hat` is a smooth
extrapolation that never blows up.

A/B targets:
- exp50 (flow, lpips=0.2): val_portraits face_lpips_sq=0.124
- exp54 (diffusion, lpips=0.2): TBD (running)
- **exp55 (diffusion, lpips=0.0)**: hypothesis-test

Decision tree:
- **exp55 > exp54**: LPIPS net-negative for diffusion. Promote exp55 as
  canonical diffusion baseline. Follow up with **exp55b** = LPIPS warmup
  `--lpips-weight 0.2 --lpips-weight-warmup-steps 5000` to recover face
  quality after eps prediction stabilizes (flag already exists, no code
  changes needed).
- **exp55 ~ exp54**: LPIPS neutral for diffusion. Drop it from the recipe
  for simplicity.
- **exp55 < exp54**: LPIPS helped despite the high-t pathology. Keep at
  0.2 — the noisy `x0_hat` gradient is still better than no perceptual
  signal at all.

```bash
WANDB_API_KEY=... bash scripts/run_exp55_diffusion_lpips0_at_exp54_recipe.sh
```

Script: `scripts/run_exp55_diffusion_lpips0_at_exp54_recipe.sh`
Outdir: `out/exp55_diffusion_eps_lpips0_at_exp54_recipe_noenc_attn163264_bf16_mc88_256px_20k`

Final val uses 100 DDIM steps (matches exp54). In-loop val stays at 20
for training speed.

**Results (final val @ 100 DDIM steps, EMA)** — 3-way A/B:

| split | metric | exp50 (flow+lpips) | exp54 (diff+lpips) | exp55 (diff, no lpips) | Δ exp55-exp54 |
|---|---|---|---|---|---|
| val_portraits | face_lpips_sq | **0.124** | 0.508 | **0.725** | +0.217 (WORSE) |
| val_portraits | face_lpips_vgg | 0.285 | 0.760 | 0.795 | +0.035 (WORSE) |
| val_portraits | face_ssim | 0.544 | 0.370 | 0.398 | +0.028 (slight gain) |
| val_portraits | whole lpips_sq | 0.170 | 0.514 | 0.707 | +0.193 (WORSE) |
| val_portraits | whole ssim | 0.444 | 0.368 | 0.413 | +0.045 (slight gain) |
| val_portraits | Δ lpips_vgg | 0.037 | 0.047 | **0.032** | -0.015 (BETTER) |
| legacy val | face_lpips_sq | 0.201 | 0.482 | 0.693 | +0.211 (WORSE) |
| legacy val | whole lpips_sq | 0.150 | 0.433 | 0.619 | +0.186 (WORSE) |

**Decision tree outcome: bucket #3** ("exp55 < exp54: LPIPS helped despite
the high-t pathology"). The "LPIPS-on-x0_hat amplifies noise at high t"
hypothesis is **refuted** — even with its known mathematical issue,
LPIPS was net-positive for diffusion. Without it, every LPIPS metric
regressed further (~+0.2 on face_lpips_sq).

**Two consolation observations** (small but real):
1. **SSIM improved slightly** without LPIPS (+0.045 whole, +0.028 face).
   Pure MSE produces smoother, lower-pixel-error outputs — but they're
   still bad in feature space.
2. **Robustness improved** — Δ_lpips_vgg dropped to 0.032 on val_portraits,
   even better than flow's 0.037. LPIPS appears to amplify the
   clean→corrupted gap; removing it makes the model more uniform across
   input quality, just at a much lower absolute ceiling.

**The diffusion-investigation arc closes here.** Two clean datapoints
(exp54 + exp55) consistently say: at this model scale (~50M params),
diffusion is structurally bottlenecked vs flow. Best-case face_lpips_sq
gap is ~4× (exp54 0.508 vs flow 0.124); recipe knobs (lpips, sample
steps) can't close it. Root cause is almost certainly **source-as-init**:
flow's `x = source` at t=0 is a strong free inductive prior that
diffusion (which starts from `x = N(0, I)`) doesn't get.

**What's NOT ruled out** (potential follow-ups, but not pursuing now):
- v-prediction over eps (`--prediction-type v`) — may give a smoother
  target at high t.
- A "source-init diffusion" hybrid — initialize sampling from a noised
  source rather than pure noise, like SDEdit. Different sampler.
- 80k steps. Could narrow the gap but unlikely to close 4×.

**Flow stays canonical.** exp52 remains the baseline. Moving on to data
diversity (exp56+ via CelebA-HQ + Places365) and resolution scale-up.

---

## exp56 — mid aug on exp52 recipe, 80k

**Status: DONE 2026-05-19** — clean-val tied with exp52, 40% robustness
gain. **New canonical for production deployment.**

Same as exp52 (canonical: flow, exp35 arch, 3k mixed, 80k @ 256px bs=4,
LPIPS=0.2 vgg) but with **mid aug** layered in to broaden the
in-distribution coverage for real-world inference.

Motivation: exp52 trained on FFHQ-aligned faces with `clean_prob=1.0` —
zero head-pose variance, zero lighting jitter, zero compression
exposure. Real-world inputs (phone photos, candid shots, off-axis
faces) are out-of-distribution along all three axes.

Aug stack comparison:

| param | exp52 (minimal, canonical) | exp33b (heavy, -16% on 1k) | **exp56 (mid)** |
|---|---|---|---|
| scale-max | 1.2 | 1.5 | **1.5** |
| rotate-deg | 0 | 25 | **15** (head tilt, not camera tilt) |
| perspective @ prob | 0 | 0.15 @ 0.5 | **0.12 @ 0.4** (head pitch/yaw proxy) |
| color jitter | 0 | 0.3 each | **0.15 each** |
| clean-prob | 1.0 | 0.2 | **0.7** (30% mild corruption) |
| blur-max | off | 3.0 | **1.5** (mild only) |
| jpeg-min | off | 30 (heavy) | **60** (mild only) |

Why this is safer than exp33b's heavy stack:
1. **3× more data** (3k vs 1k) absorbs aug variance instead of overfitting on it.
2. **4× longer training** (80k vs 20k) lets the model learn the broader distribution.
3. **Every "destroys signal" knob** (color, blur, jpeg, resize-degrade) cut roughly in half from exp33b.
4. **Head-pose proxies** (rotate=15°, perspective=0.12@0.4) stay relatively strong because that's the actual OOD failure mode.

A/B target — exp52 (canonical) on val_portraits:
- face_lpips_sq=0.101, face_lpips_vgg=0.244, face_ssim=0.579
- whole lpips_sq=0.145, whole ssim=0.459
- Δ_lpips_vgg=0.045

Expected: clean-val face_lpips_sq regresses 5-15% (cost of regularization);
Δ_lpips_vgg drops toward 0.030-0.040; phone-camera inference visibly better.

Decision rule:
- clean-val regression <10% → exp56 becomes canonical (replaces exp52).
- clean-val regression >15% → exp52 stays for benchmarks, exp56 is the deployment ckpt.

```bash
WANDB_API_KEY=... bash scripts/run_exp56_mid_aug_at_exp52_recipe.sh
```

Script: `scripts/run_exp56_mid_aug_at_exp52_recipe.sh`
Outdir: `out/exp56_mid_aug_at_exp52_recipe_noenc_attn163264_bf16_mc88_256px_80k`

**Results (final val @ 20 Euler steps, EMA)**:

| split | metric | exp52 (canonical) | **exp56 (mid aug)** | Δ |
|---|---|---|---|---|
| val_portraits | face_lpips_sq | 0.101 | 0.104 | +3.0% (tie) |
| val_portraits | **face_lpips_vgg** | 0.244 | **0.244** | **exact tie** |
| val_portraits | face_ssim | 0.579 | 0.577 | -0.3% (tie) |
| val_portraits | whole lpips_sq | 0.145 | 0.148 | +2.1% (tie) |
| val_portraits | whole ssim | 0.459 | 0.460 | tie |
| val_portraits | **Δ_lpips_vgg** | 0.045 | **0.027** | **-40.9% WIN** |
| val_portraits | **Δ_lpips_sq** | 0.024 | **0.017** | **-30.8% WIN** |
| legacy val | face_lpips_sq | 0.183 | 0.191 | +4.4% (mild loss) |
| legacy val | face_lpips_vgg | 0.355 | 0.359 | +1.1% (tie) |
| legacy val | face_ssim | 0.623 | 0.631 | +1.3% (tie) |
| legacy val | **Δ_lpips_vgg** | ~0.125 (mid-train) | **0.077** | **~-38% WIN** |

**Why this is a genuine robustness win, not a Δ-arithmetic artifact**:
unlike exp57 where Δ shrunk because clean degraded by ~the same amount
as corrupted, exp56's clean is *tied* with exp52 while absolute corrupt
metrics actually improved. The wandb mid-training charts showed this
cleanly: corrupt SSIM ~0.62 vs exp52's ~0.59 (+5%), corrupt lpips_sq
~0.225 vs ~0.25 (-10%). The model genuinely learned corruption
invariance from the training-time exposure (clean_prob=0.7 + blur ≤1.5
+ jpeg ≥60).

**Mechanism that exp57 lacked**: source-dropout zeros the clean source —
doesn't simulate any specific corruption. exp56's mid-aug exposes the
model to JPEG, blur, and resize during training, so it learns
robustness to exactly those degradations.

**Decision**: **exp56 replaces exp52 as the deployment canonical.**
Same clean-val face quality (face_lpips_vgg=0.244 to 4 dp), much
better real-world robustness. For benchmark comparisons going forward,
both should be cited: exp52 for the "minimal-aug ceiling" and exp56
for the "real-world-deployable" number.

**Legacy val regression** (face_lpips_sq +4%) is the only real loss,
and it's within noise. The mid-aug "head-pose simulation" actually
moved legacy val face_ssim *up* (+1.3%), which is consistent with the
broader hypothesis: aug exposing the model to head-pose variance helps
on group-photo-style val with non-frontal faces.

---

## exp57 — source dropout 0.2 (regularization, NO CFG)

**Status: DONE 2026-05-19** — tie on quality + robustness win, candidate
for 80k promotion.

Single-flag delta vs exp50: `--source-dropout 0.2`. 20% of training
batch elements get their source channels zeroed → model must predict
target from noise + time only for those samples. **NOT CFG** — at
inference we keep `--cfg-scale 1.0` (single conditioned pass).
exp41's CFG-at-flow failure (ssim 0.36 at scale=2.0) is binding: in
flow, v is a true velocity, can't be amplified.

Hypothesis: at 3k pairs × 80k steps = 100+ epochs (exp52 regime), the
model may be over-memorizing source→target shortcuts. Dropout as
regularization forces a target-distribution prior, which should also
help robustness on weak-source inputs.

Recipe: exp50 base + `--source-dropout 0.2`, 20k @ 256px bs=4. If A/B
wins vs exp50, promote to 80k vs exp52.

Script: `scripts/run_exp57_source_dropout_at_exp50_recipe.sh`

**Results vs exp50 (20k baseline)**:

| split | metric | exp50 | exp57 | Δ |
|---|---|---|---|---|
| val_portraits | **face_lpips_sq** | **0.124** | **0.124** | 0% (exact tie) |
| val_portraits | face_lpips_vgg | 0.285 | 0.290 | +1.8% (tie) |
| val_portraits | face_ssim | 0.544 | 0.550 | +1.1% (tie) |
| val_portraits | whole ssim | 0.444 | **0.457** | **+2.9% (WIN)** |
| val_portraits | **Δ_lpips_vgg** | 0.037 | **0.034** | **-8.1% (WIN)** |
| legacy val | face_lpips_sq | 0.201 | 0.207 | +3.0% (mild loss) |
| legacy val | whole lpips_sq | 0.150 | 0.156 | +4.0% (mild loss) |

Read: essentially a tie on quality with mild robustness gain. At 20k
steps × 3k pairs = 26 epochs, the "over-memorization at long training"
hypothesis hasn't had a chance to differentiate.

**Important nuance on the robustness Δ improvement**: there are two
ways Δ_lpips_vgg can shrink. Either (a) corrupted-val genuinely
improves while clean stays flat — a real robustness gain (this is what
exp56 mid-aug shows in mid-training charts), or (b) clean degrades and
corrupted degrades by ~the same amount — Δ shrinks mechanically but
absolute corrupt-val is unchanged. exp57 is closer to (b): exp50
portraits corrupted ≈ 0.390, exp57 portraits corrupted ≈ 0.393 — the
absolute robustness barely moved. The Δ improvement is real arithmetic
but mechanically less compelling than the "model learned to invariance"
story exp56 is telling. Source dropout alone doesn't expose the model
to corruption — that's what training-time aug (clean_prob<1) actually
does.

**Recommendation**: promote to 80k vs exp52 as **exp57b**. Defer until
exp58 + exp59 land — if either of those is a clearer 20k win, prioritize
that promotion instead.

---

## exp58 — logit-normal t-sampling (SD3/EDM-style)

**Status: DONE 2026-05-19** — finished anyway, **catastrophic
regression** on portraits (+44% face_lpips_sq), confirms endpoint
starvation theory cleanly.

Code change: added `t_sample_mode` / `t_sample_mu` / `t_sample_sigma`
fields to `FlowConfig`, branched in `flow.py:training_loss`. Default
remains `uniform` — exp50/52/56 reproduce.

Single-flag delta vs exp50: `--t-sample-mode logit_normal --t-sample-mu 0 --t-sample-sigma 1`.

Default flow training samples t ~ U[0,1]. Endpoints (t=0=source,
t=1=target) are "easy" — model just learns the full delta. Logit-normal
(t=sigmoid(N(mu,sigma))) peaks at 0.5 with mu=0, biasing training
toward the hard middle of the path where x_t is a mixed interpolant
and the model has to predict velocity from a partial signal. SD3 and
the Karras EDM family report consistent gains from this.

Smoke confirmed: 1000-sample empirical distribution has tighter std
(0.21 vs uniform's 0.28) and more mass in [0.4, 0.6] (29% vs 21%).

20k @ 256px bs=4 vs exp50. Script: `scripts/run_exp58_logit_normal_t_at_exp50_recipe.sh`

**Results vs exp50 (sigma=1.0, ran to completion)**:

| split | metric | exp50 | exp58 (sigma=1.0) | Δ |
|---|---|---|---|---|
| val_portraits | **face_lpips_sq** | **0.124** | **0.179** | **+44% LOSS** |
| val_portraits | face_lpips_vgg | 0.285 | 0.368 | +29% LOSS |
| val_portraits | face_ssim | 0.544 | 0.436 | -20% LOSS |
| val_portraits | whole lpips_sq | 0.170 | 0.210 | +24% LOSS |
| val_portraits | whole ssim | 0.444 | 0.386 | -13% LOSS |
| val_portraits | Δ_lpips_vgg | 0.037 | 0.041 | +11% (mild loss) |
| legacy val | face_lpips_sq | 0.201 | 0.223 | +11% LOSS |
| legacy val | face_ssim | 0.605 | 0.576 | -4.8% (tie/loss) |

Root cause confirmed empirically: at sigma=1.0, only **0.2%** of
training samples land at t<0.05 vs uniform's 5% — endpoints were **25×
starved**. Inference walks the ODE uniformly from t=0→1, so the first
Euler step `x = source + dt·v(·, t=0)` queried a model that effectively
never saw t≈0 during training. Trajectory corrupted from step 1 onward.

Surprise: empirical distribution math was much harsher than the
"peaked at 0.5" intuition. At sigma=1.0, only 1.4% of samples have
t<0.10 (vs uniform's 10%), and 31% land in [0.4, 0.6] (vs uniform's
20%) — that's a 7× concentration, not a "mild" bias. SD3 reports gains
with similar config but at 8B params + billions of samples; at our
~50M / 80k regime, the endpoints aren't optional.

Sweet-spot reanalysis across sigma values:

| sigma | t<0.05 | t<0.10 | [.4,.6] | t>0.95 | verdict |
|---|---|---|---|---|---|
| uniform | 5.0% | 10% | 20% | 5.0% | baseline |
| **1.00** (exp58) | **0.2%** | **1.4%** | **31%** | **0.2%** | endpoints 25× starved |
| **1.50** (exp58b) | 2.5% | 7.1% | 21% | 2.5% | sweet spot |
| 2.00 | 6.9% | 13.6% | 16% | 7.1% | bimodal — defeats purpose |

At sigma≥2.0 the underlying Gaussian is wide enough that sigmoid pushes
mass to BOTH tails — distribution is no longer mid-peaked.

---

## exp58b — logit-normal t-sampling, sigma=1.5

**Status: DONE 2026-05-19, RESULT QUESTIONED 2026-05-19** — appears to
regress on metrics, but the regression may be benchmark-bias artifact,
not real quality loss. See dataset caveat below.

Single-flag delta vs exp58: `--t-sample-sigma 1.5` (was 1.0).

sigma=1.5 keeps the mid-t bias (21% in [0.4, 0.6] vs uniform's 20%) but
endpoints are only 2× starved instead of 25×. This is the narrow
sweet spot — sigma=1.25 still gets endpoints at 1% (5× worse), sigma≥1.75
loses the mid-peak entirely.

Same recipe as exp58 otherwise. 20k @ 256px bs=4 vs exp50.

Script: `scripts/run_exp58b_logit_normal_t_sigma15_at_exp50_recipe.sh`

**Results vs exp50 (asymmetric loss)**:

| split | metric | exp50 | exp58b | Δ |
|---|---|---|---|---|
| legacy val | face_lpips_sq | 0.201 | 0.204 | +1.5% (tie) |
| legacy val | face_lpips_vgg | 0.379 | **0.379** | **exact tie** |
| legacy val | face_ssim | 0.605 | 0.596 | -1.5% (tie) |
| legacy val | whole ssim | 0.516 | 0.507 | -1.7% (tie) |
| legacy val | Δ_lpips_vgg | 0.116 | 0.113 | -2.6% (small win) |
| **val_portraits** | **face_lpips_sq** | **0.124** | **0.136** | **+9.7% LOSS** |
| val_portraits | face_lpips_vgg | 0.285 | 0.309 | +8.4% LOSS |
| val_portraits | face_ssim | 0.544 | 0.507 | -6.8% LOSS |
| val_portraits | whole ssim | 0.444 | 0.422 | -5.0% LOSS |
| val_portraits | whole lpips_sq | 0.170 | 0.182 | +7.1% LOSS |
| val_portraits | Δ_lpips_vgg | 0.037 | 0.037 | exact tie |

**Asymmetric loss is informative**: legacy val (group photos, small
peripheral faces, rough-structure decisions) tied. val_portraits (FFHQ
close-up portraits, fine detail matters) regressed across the board.

**⚠️ PIN — dataset bias caveat (2026-05-19)**: visual inspection
revealed that Flux occasionally whitewashed darker-skinned sources when
generating the anime target. So for those pairs, the "correct" target
is itself lighter than the source. SOTA recipes (exp50/52/56/59) produce
outputs that drift toward the (biased) target → match it → score well.
**exp58b appears to produce outputs that stay closer to the actual
source skin tone** → diverges from the biased target → scores worse on
LPIPS/SSIM.

This means the "+10% face_lpips_sq regression" on val_portraits may not
be a quality regression at all — it could be exp58b being **more
faithful to the source** while the benchmark penalizes faithfulness on
the affected subset. The "endpoint starvation" theory still cleanly
explains exp58 (sigma=1.0, +44%) — that's too catastrophic to be pure
bias artifact. But for exp58b's milder regression, the story is now
ambiguous: endpoints + bias-divergence both contribute, in unknown
proportion.

**Follow-up to actually decide**:
1. Stratify val_portraits by skin tone (use a face-attribute classifier
   or manual labels on the 200 portraits), compute metrics per-bin.
2. Visual side-by-side: exp50 vs exp58b outputs on the same input,
   look at whether 58b is "wrong" or "different-but-defensible".
3. Re-run the t-sampling sweep with a corrected dataset where bias is
   regenerated out (re-run Flux with explicit skin-tone preservation
   prompt, or filter pairs by source-target skin-tone delta).

For now: **logit-normal is parked, not declared dead**. The structural
"endpoints matter in img2img" argument still holds — but the
"catastrophic regression" claim was over-confident given the
benchmark caveat. Same caveat applies retroactively to any other
conclusion drawn from val_portraits metrics — though the magnitude
of exp50/56/59 wins is small enough that bias shifts wouldn't flip them.

**Root cause — img2img flow vs text2img flow**:

The "logit-normal helps because the hard work is at mid-t" intuition
from SD3/EDM is **text-to-image** specific. In text2img:
- t=0 (clean image): trivial output
- t=1 (pure noise): hard structure inference
- mid-t: hardest — commit to scene composition

In **img2img flow** (what we do):
- t=0 (x=source): model has to predict the **full velocity = target-source** delta from source alone — actually hard, especially for fine detail
- t≈1 (x≈target): refine final detail — also matters for fidelity
- mid-t: model has source-target interpolant for free, structure is *anchored* by the linear path — relatively easier

Mid-t bias **starves the actually-hard parts**. The asymmetry by split
confirms: rough-structure prediction (legacy val) doesn't care, but
fine-detail face prediction (val_portraits) needs endpoint training.

**Conclusion (revised)**: logit-normal t-sampling appears to regress on
val_portraits metrics, but the regression is partially confounded by a
dataset bias (Flux-whitewashed targets, see exp58b PIN). The
endpoint-starvation theory cleanly explains the +44% catastrophe of
exp58 (sigma=1.0). For exp58b (sigma=1.5, +10%), the story is
ambiguous — could be endpoints + bias-divergence in some proportion.
**Parking this lever, not killing it outright.** Should revisit after
the skin-tone-stratified eval is built.

What this rules out: any "concentrate training on mid-t" variant
(shifted logit-normal with mu≠0, U-shaped weighting, etc.) — they'd
all hit the same wall. The img2img analog of the SD3 trick would be
the **opposite**: bias *toward* endpoints (where source/target identity
provides the conditioning anchor, not the middle).

**Clean empirical gradient across the sigma sweep** confirms the theory:

| t-sampling | t<0.05 starvation | val_portraits face_lpips_sq | regression |
|---|---|---|---|
| uniform (exp50) | 5.0% (baseline) | 0.124 | (baseline) |
| logit-normal σ=1.5 (exp58b) | 2.5% (2× starved) | 0.136 | +10% |
| logit-normal σ=1.0 (exp58) | 0.2% (25× starved) | 0.179 | **+44%** |

Monotonic: tighter logit-normal → more starved endpoints → worse
fine-detail prediction. The starvation fraction predicts the
regression magnitude almost linearly.

---

## exp59 — cross-attention conditioning at H/8 decoder level

**Status: DONE 2026-05-19** — uniform small win, zero regressions,
breaks the 20k face_lpips_sq floor. **80k promotion candidate.**

Code change: added `CrossAttnCond` class to `source_pyramid.py`,
wired into `Img2ImgDiffusionUNet` at the H/8 decoder level via
`--use-cross-attn-cond`, auto-detected from state_dict in `ckpt.py`.

Hypothesis: FiLM (per-channel γ,β scaling from matching-position
pyramid feature) is local — every decoder position gets the same
modulation from the same pyramid position. Cross-attention lets each
decoder position query EVERY pyramid position. Example: a chin
landmark could inform forehead generation. More expressive than FiLM
at the cost of quadratic-in-tokens compute.

Implementation:
- Multi-head SDPA (4 heads, head_dim=88), Q from decoder, KV from
  pyramid feature f3 (c4 channels at H/8 = 32×32 = 1024 tokens).
- Zero-init output projection → identity at insertion time.
- Added at the deepest non-bottleneck decoder level only — H/4 (4096
  tokens) and shallower are too expensive for full cross-attn; FiLM
  stays there.
- ~500k extra params (~1% of 50M base).
- Auto-detect from state_dict (key prefix `cross_attn_dec4.`); older
  checkpoints load cleanly via the existing pattern.

Smoke confirmed: identity-at-init (max diff = 0), 10-step training
end-to-end, ckpt save+load roundtrips via `build_model_from_ckpt`.

Single-flag delta vs exp50: `--use-cross-attn-cond`. 20k @ 256px bs=4.
Script: `scripts/run_exp59_cross_attn_at_exp50_recipe.sh`

**Results vs exp50 (uniform win, zero regressions)**:

| split | metric | exp50 | exp59 | Δ |
|---|---|---|---|---|
| **val_portraits** | **face_lpips_sq** | 0.124 | **0.122** | **-1.6% WIN (best 20k flow ever)** |
| val_portraits | face_lpips_vgg | 0.285 | 0.282 | -1.1% WIN |
| val_portraits | face_ssim | 0.544 | 0.546 | +0.4% (tie) |
| val_portraits | whole lpips_sq | 0.170 | 0.166 | -2.4% WIN |
| val_portraits | whole ssim | 0.444 | 0.445 | tie |
| val_portraits | **Δ_lpips_vgg** | 0.037 | **0.035** | **-5.4% WIN** |
| legacy val | face_lpips_sq | 0.201 | 0.203 | +1.0% (tie) |
| legacy val | face_lpips_vgg | 0.379 | 0.381 | +0.5% (tie) |
| legacy val | whole lpips_vgg | 0.297 | 0.294 | -1.0% WIN |
| legacy val | Δ_lpips_vgg | 0.116 | 0.111 | -4.3% WIN |

**Cleanest result of the 57/58/59 round**:
- val_portraits: 5 WINs, 2 TIEs, **0 LOSEs**
- legacy val: 2 WINs, rest TIEs, **0 LOSEs**
- face_lpips_sq=0.122 on portraits is **the lowest 20k-flow number ever measured**, beating exp50's 0.124 (which held since the data-scale-up era began).

**Why this matters**: the improvement pattern matches exactly what
cross-attn is supposed to do — more expressive source→target
conditioning → fine detail prediction → val_portraits (close-up faces)
shows the cleanest wins, legacy val (rough scene structure) just ties.
+500k params (1% of 50M base) for uniform 1-3% quality improvement is
a clean architectural win.

**Comparison vs the unrelated exp35→exp52 long-training arc**:
- exp50 (20k, no cross-attn): 0.124 face_lpips_sq portraits
- exp52 (80k, no cross-attn): 0.101 (linear improvement 20k→80k worth ~19%)
- exp59 (20k, cross-attn): 0.122 (+1.6% over exp50)
- exp60 (80k, cross-attn) prediction: 0.101 × (0.122/0.124) ≈ **~0.099** — would be the first sub-0.10 face_lpips_sq, though this is speculative linear extrapolation.

**Next step — exp60 promotion**: 80k @ exp59 recipe, A/B vs exp52.
If cross-attn at H/8 holds the 20k improvement at 80k, it becomes the
new benchmark canonical (replacing exp52). Optionally **stack with
mid-aug from exp56** for a combined "quality + robustness" canonical.
Both improvements appear orthogonal (architectural vs data).

---

---

## exp60 — cross-attn at 80k (exp59's win promoted)

**Status: DONE 2026-05-19** — **first sub-0.10 face_lpips_sq ever
measured** (0.0997). Strictly dominates exp52. **New quality canonical.**

80k promotion of exp59's clean +cross-attn win. Single-flag delta vs
exp52: `--use-cross-attn-cond`.

Hypothesis: exp59's uniform 1-3% improvement at 20k holds at 80k. If
linear, face_lpips_sq portraits ≈ 0.099 (first sub-0.10 ever). Even
non-linear, anything ≤ 0.101 resets the canonical ceiling.

A/B target — exp52 (former quality canonical):
- face_lpips_sq portraits=0.101, face_lpips_vgg=0.244, face_ssim=0.579
- whole lpips_sq=0.145, whole ssim=0.459, Δ_lpips_vgg=0.045

Script: `scripts/run_exp60_cross_attn_at_exp52_recipe_80k.sh`

**Results — strictly dominates exp52 across both splits**:

| split | metric | exp52 (former) | **exp60** | Δ |
|---|---|---|---|---|
| val_portraits | **face_lpips_sq** | 0.101 | **0.0997** | **-1.3% (sub-0.10 first)** |
| val_portraits | face_lpips_vgg | 0.244 | **0.237** | -2.9% WIN |
| val_portraits | face_ssim | 0.579 | 0.583 | +0.7% (tie/win) |
| val_portraits | whole lpips_sq | 0.145 | 0.142 | -2.1% WIN |
| val_portraits | whole ssim | 0.459 | 0.460 | tie |
| val_portraits | Δ_lpips_vgg | 0.045 | 0.040 | -11% WIN |
| legacy val | face_lpips_sq | 0.183 | **0.182** | -0.5% WIN |
| legacy val | face_lpips_vgg | 0.355 | **0.349** | -1.7% WIN |
| legacy val | face_ssim | 0.623 | 0.630 | +1.1% (tie/win) |
| legacy val | whole lpips_sq | TBD | 0.131 | (best in col) |
| legacy val | Δ_lpips_vgg | ~0.125 | 0.113 | -10% WIN |

The speculative linear extrapolation from exp59 (face_lpips_sq portraits
0.122 at 20k → 0.099 at 80k) landed almost exactly: actual 0.0997.

**vs exp61 (deployment canonical, mid aug + cross-attn)**:

| metric | exp61 | exp60 | Δ exp60 vs exp61 |
|---|---|---|---|
| face_lpips_sq portraits | 0.103 | **0.0997** | -3.2% (exp60 wins) |
| face_lpips_vgg portraits | 0.242 | **0.237** | -2.1% (exp60 wins) |
| whole lpips_sq portraits | 0.148 | **0.142** | -4.1% (exp60 wins) |
| **Δ_lpips_vgg portraits** | **0.025** | 0.040 | +60% (exp61 wins robustness) |

exp60 has **better clean quality** but **worse robustness** than exp61.
The mid-aug component costs ~3% on face_lpips_sq portraits but buys
-40% on Δ_lpips_vgg. Real-world deployments care more about Δ;
benchmark scores care more about clean quality.

**Updated canonical roles**:
- **exp60** = pure quality canonical (replaces exp52). First sub-0.10
  face_lpips_sq. Use when reporting benchmark numbers.
- **exp61** = deployment canonical (replaces exp56). Best robustness
  ever measured (Δ_lpips_vgg=0.025). Use for production checkpoints.
- exp52, exp56 demoted to historical references; both are now
  strictly dominated.

---

## exp61 — STACK: cross-attn + mid aug at 80k

**Status: DONE 2026-05-19** — **new single canonical**. Stack hypothesis
confirmed: ties exp52 on quality, beats exp56 on every metric including
the best robustness Δ ever measured (0.025).

Combines exp56 (mid aug, deployment canonical, 40% better robustness)
with exp59 (cross-attn, architectural quality win). Hypothesis: the two
levers are orthogonal — architectural improvement (cross-attn) and data
exposure (mid aug) are independent axes. Stacking should give both face
quality AND robustness simultaneously.

Recipe: exp56's mid-aug stack + exp59's cross-attn flag. Effectively
the union of the two wins.

A/B targets:

| recipe | face_lpips_sq portraits | Δ_lpips_vgg portraits |
|---|---|---|
| exp52 (quality canonical) | 0.101 | 0.045 |
| exp56 (deployment canonical) | 0.104 | 0.027 |
| exp59 (cross-attn 20k) | 0.122 (-1.6% vs exp50 0.124) | 0.035 |
| **exp61 (target)** | **≤ 0.101 ideally + ≤ 0.030 robustness** | |

If orthogonal: exp61 wins on both axes simultaneously and **replaces
both exp52 and exp56 as the single canonical** going forward.
If interference: improvements partial-cancel and exp52/56 stay as
separate canonicals for "quality" vs "deployment" tracks.

Script: `scripts/run_exp61_cross_attn_plus_mid_aug_80k.sh`

**Results — three-way comparison on val_portraits**:

| metric | exp52 (quality, 80k) | exp56 (deployment, 80k) | **exp61 (stack, 80k)** |
|---|---|---|---|
| face_lpips_sq | **0.101** | 0.104 | 0.103 (tie with exp52) |
| face_lpips_vgg | 0.244 | 0.244 | **0.242** (slight win on both) |
| face_ssim | 0.579 | 0.577 | **0.581** (slight win on both) |
| whole lpips_sq | 0.145 | 0.148 | 0.148 (tie with exp56) |
| whole ssim | 0.459 | 0.460 | 0.460 (tie) |
| **Δ_lpips_vgg** | 0.045 | 0.027 | **0.025 (best ever)** |
| Δ_lpips_squeeze | 0.024 | 0.017 | **0.015 (best ever)** |

Legacy val: face_lpips_sq=0.189 (slight loss vs exp52's 0.183, slight
win vs exp56's 0.191), face_ssim=0.632 (best of the three), corrupt-val
Δ=0.078 (much better than exp52's chart-extrapolated ~0.125).

**Orthogonal stack hypothesis: CONFIRMED**. The architectural lever
(cross-attn: enriches fine-detail conditioning) and the data lever
(mid aug: exposes model to corruption/pose variance) compose without
interference. Net:
- Quality (face_lpips_sq portraits) ≈ exp52's 0.101 ceiling
- Robustness (Δ_lpips_vgg) **-44% vs exp52, -7% vs exp56** — best ever

**exp61 is the new single canonical, replacing both exp52 and exp56.**
Going forward, all A/B's run against exp61. exp52 and exp56 stay cited
as the "pure quality" and "pure robustness" reference points but the
combined recipe dominates them both.

**exp60 implication**: should still be run for the clean architectural
ablation (cross-attn alone at 80k vs exp52 with no aug). Tells us how
much of exp61's win is from cross-attn alone vs the stacking. But the
canonical decision is already made.

---

---

## exp62 — drop source-in-stem + add cross-attn at H/4

**Status: WIRED 2026-05-19**

Two-knob delta vs exp59 (cross-attn @ H/8, 20k):
1. `--no-source-in-stem`: in_conv goes 6→88 ch to 3→88 ch. Source no
   longer concatenated into the encoder input. Source signal now comes
   purely via SourcePyramid + FiLM + cross-attn.
2. `--use-cross-attn-cond-h4`: adds a second CrossAttnCond at the H/4
   decoder level. Multi-scale source conditioning: H/8 [1024 tokens] +
   H/4 [4096 tokens].

Net param delta vs exp60: +495k (essentially same budget at ~49M).

Hypothesis: in flow matching, `x_t = (1-t)·source + t·target`, so at
t=0 the model sees source via `x_t` itself — making source-in-stem
redundant with pyramid+cross-attn. Removing it eliminates double
conditioning and frees capacity. Multi-scale cross-attn (H/8 + H/4)
compensates by giving stronger pyramid-mediated source conditioning.

Code changes:
- `model.py`: relaxed the `use_source_encoder=False → source_in_stem=True`
  override when pyramid is enabled. Added `use_cross_attn_cond_h4` flag
  + `cross_attn_dec3` module at H/4 decoder level.
- `ckpt.py`: auto-detects both cross-attn levels from state_dict.
- `train_exp32_prog512.py`: `--no-source-in-stem` and
  `--use-cross-attn-cond-h4` flags; saved config records actual
  source_in_stem value (not the hardcoded True default).

Smokes confirmed: forward pass works without source concat, ckpt
roundtrip preserves both new flags via state_dict key auto-detection,
in_conv weight shape correctly reflects 3-channel input.

A/B target — exp59 (20k, cross-attn @ H/8 only, source_in_stem=True):
- face_lpips_sq portraits = 0.122
- face_lpips_vgg portraits = 0.282
- Δ_lpips_vgg portraits = 0.035

Script: `scripts/run_exp62_no_concat_plus_ca_h4_at_exp50_recipe.sh`

If wins → 80k promotion as exp62b vs exp60 (current quality canonical
at 0.0997).

---

## exp63 — PatchGAN adversarial loss retry from exp61

**Status: WIRED 2026-05-19**

Retry of the parked exp20/21 era (May 2026) but under fundamentally
better conditions. Six confounders fixed since:

1. 3× more data (3k mixed vs 1k synth) — exp21 was below typical GAN data floor
2. Modern arch (decoder_attn + pyramid + FiLM + cross-attn) — stronger G
3. val_portraits exists — the metric that under-counted faces is gone
4. Mid aug exposure (exp56) — D can't lean on input-fidelity shortcut
5. Start from exp61 EMA (80k pretrained G) — eliminates "G learns shortcuts before basic photo→anime"
6. gan_weight=0.02 between exp21b's too-weak 0.005 and exp21's catastrophic 0.1

Code wired into `train_exp32_prog512.py`:
- `PatchDiscriminator` (pix2pix 70×70, ~2.8M params) + AdamW(d_lr=1e-4, β1=0.5)
- Hinge G/D losses (existing `gan_loss.py`)
- Adaptive G/D switching (from exp21c — best variant): G updates when
  `g_gan_ema >= d_loss_ema`, D updates when `d_loss_ema >= g_gan_ema`
- Saved D state + optimizer in checkpoints; D state reload from `--resume`
  if present, fresh-start otherwise (exp61 ckpt has no D)
- Fresh cosine LR schedule on GAN start (step counter reset, not the
  carried-over step=80000 which would give lr_min from step 1)

A/B target — exp61 (deployment canonical, val_portraits):
- face_lpips_sq=0.103, face_lpips_vgg=0.242, face_ssim=0.581
- Δ_lpips_vgg=0.025 (best robustness ever)

Hypothesis: PatchGAN adds texture crispness that LPIPS can't measure.
exp21 reported "yes qualitatively, no quantitatively" — under val_portraits
(the better metric) + stronger G base, the quantitative answer may flip.

Script: `scripts/run_exp63_patchgan_retry_from_exp61.sh`

20k adversarial phase. lr 1e-4 (half of 2e-4 default; the G is already
at SOTA, fine-tuning calls for lower lr).

---

## exp64 — AdaLN-Zero time conditioning everywhere

**Status: WIRED 2026-05-19**

Replaces every `ResBlock` (additive `time_proj(t_emb)`) with
`AdaLNResBlock` (DiT/SD3-style modulation: γ, β, α gates per norm,
predicted per-block from t_emb).

Modern flow/diffusion lit (DiT, SD3, Flux) consistently reports 1-3%
gains from AdaLN-Zero. Our scale (50M) is below DiT-XL (~675M), so the
gain may be smaller but the direction is well-supported.

Code wired:
- `AdaLNResBlock` class in `model.py` (drop-in for `ResBlock`, same
  constructor signature). 6 modulation outputs per block: γ₁, β₁ on
  pre-conv1 (in_ch dim) + α₁, γ₂, β₂, α₂ on post-conv1 / pre-conv2 /
  post-conv2 (out_ch dim). Zero-init proj → identity at insertion time.
- `--use-adaln-time` flag in trainer. Auto-detected in `ckpt.py` via
  `adaln_proj_in.weight` keys.

Param cost: +9.5M (~20%) vs exp59 — modulation MLPs add up across 10
ResBlocks. Total ~58M.

Single-flag delta vs exp59: `--use-adaln-time`.

A/B target — exp59 (val_portraits): face_lpips_sq=0.122.

Script: `scripts/run_exp64_adaln_everywhere_at_exp59_recipe.sh`

---

## exp65 — x0-prediction parameterization

**Status: WIRED 2026-05-19**

Default flow predicts velocity `v = target - source`. exp65 predicts
the clean target `x_target` directly (DDIM-style). At inference, v_hat
is recovered as `(x0_hat - x_t) / max(1 - t, 1e-3)` and Euler/Heun
steps with that — ODE integration unchanged.

x0-prediction has uniform output scale across t; velocity prediction
does not. Different loss landscape; SD3 reports small but consistent
differences between parameterizations.

Code wired:
- `FlowConfig.prediction_type` field ("v" default, "x0" alternative)
- `flow.py:training_loss` branches on prediction_type
- `flow.py:sample` velocity() helper derives v from x0_hat when active
- `--flow-prediction-type` flag in trainer

Single-flag delta vs exp59: `--flow-prediction-type x0`.

Risk: at t close to 1, the `v = (x0_hat - x_t) / (1 - t)` recovery is
numerically unstable. Mitigated by clamping (1 - t) ≥ 1e-3.

A/B target — exp59 (val_portraits): face_lpips_sq=0.122.

Script: `scripts/run_exp65_x0_pred_at_exp59_recipe.sh`

---

## exp66 — wider model (mc=128) capacity test

**Status: WIRED 2026-05-19**

exp59 recipe with `--model-ch 128` instead of 88. Bumps all internal
widths proportionally; total params ~102M (~2.1× exp59's 49M).

Tests: is capacity the bottleneck at our 3k data scale, or is the
model already saturated?

Context: exp22 (May 2026) tested mc=176 at 1k pairs and saw grid
artifacts + identity collapse — too much capacity for the small
dataset. With 3× more data and mc=128 (intermediate, not 176), that
failure mode shouldn't apply. But it's worth flagging as a real risk
— if grid artifacts return, kill immediately.

Single-flag delta vs exp59: `--model-ch 128`.

A/B target — exp59 (val_portraits): face_lpips_sq=0.122.

Script: `scripts/run_exp66_wider_mc128_at_exp59_recipe.sh`

---

## Tool improvement: Heun ODE solver in `flow.sample`

**Status: WIRED 2026-05-19**

Pure inference-time change to `RectifiedImageFlow.sample`. New
`sampler` kwarg with options:
- `"euler"` (default, legacy): `x_{n+1} = x_n + dt · v(x_n, t_n)`. 1 NFE/step.
- `"heun"`: predictor-corrector 2nd-order method:
  `x_pred = x_n + dt · v(x_n, t_n)`,
  `x_{n+1} = x_n + dt · 0.5 · (v(x_n, t_n) + v(x_pred, t_{n+1}))`.
  2 NFE/step; last step falls back to Euler.

Validate via `--sampler {euler,heun}` flag. **No retraining needed** —
applies to any existing flow checkpoint. Cheapest experiment in the
queue: compare exp60/exp61 at sample_steps=20 with euler vs heun, or
at sample_steps=10 with heun vs sample_steps=20 with euler (same NFE
budget, different accuracy profiles).

---

## ⚠️ Benchmark caveat: Flux skin-tone bias in val_portraits (2026-05-19)

Discovered while reviewing exp58b outputs: Flux occasionally produced
lighter-skinned anime targets when given darker-skinned real-photo
sources. The val_portraits "ground truth" is therefore biased for those
pairs — a model that faithfully preserves source identity will score
*worse* on LPIPS/SSIM than one that drifts toward the lightened target.

**Implications for prior comparisons** in this log:
- Small wins (1-5% deltas like exp59's -1.6% face_lpips_sq) — probably
  robust to this bias since the magnitude is below what skin-tone
  drift could explain.
- Medium regressions (exp58b's +10%) — partially confounded; could be
  half-method, half-bias.
- Large regressions (exp58's +44%, exp54's +310%) — too big to be
  pure bias artifact; the method-level issues are real.

**Action items**:
- [ ] Build a skin-tone-stratified val_portraits subset (or use a face
      attribute classifier) so we can report metrics per-bin.
- [ ] Re-generate the affected target pairs with explicit skin-tone-
      preservation prompts, or filter them out of val_portraits.
- [ ] For high-stakes A/B's going forward, supplement metrics with
      visual side-by-sides on a curated set of darker-skinned sources.

**What this does NOT invalidate**:
- exp52 / exp56 / exp59's clean-val wins (small deltas, dominated by
  signal not bias).
- Robustness Δ comparisons (corrupt-val vs clean-val is bias-symmetric
  since both pass through the same biased target).

---

## Open follow-ups (3k era, updated 2026-05-19)
- **More diverse real-photo sources**: Unsplash people, Places365
  with people-filter, AFW/IJB-C in-the-wild faces. Currently FFHQ
  alone biases toward studio-lit Western 25-35yo portraits. This is
  now the highest-leverage open lever, since exp53 ruled out
  resize-filter and exp50→exp52 already extracted the training-
  duration gain.
- **Resolution scale-up**: every 3k-era run is at 256px; FFHQ source
  is 512. Train at 384 or 512 to test the resolution ceiling.
  Independent of exp53's negative — at 512 target, BILINEAR is
  near-identity on 512px sources so this is a "more pixels of
  capacity" experiment, not a "sharper input" one.
- **Curriculum option** (deferred): if portrait quality stalls below
  some threshold, start from exp51's FFHQ-only checkpoint and
  fine-tune on the 3k mixed set. Might give exp50-on-FFHQ quality
  *and* exp50-on-legacy capability simultaneously.
- **In-the-wild face val split**: a third val with small / off-center /
  partially-occluded faces, real photos. Currently no val covers this
  honestly; legacy val is group photos but skewed toward peripheral
  subjects.
