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

## Open follow-ups (3k era)

- **exp52 = exp50 × 80k steps** — natural next move now that data
  scale-up worked. exp25-80k extension was the legacy-era big lift;
  same pattern at the new scale should compound.
- **More diverse real-photo sources**: Unsplash people, Places365
  with people-filter, AFW/IJB-C in-the-wild faces. Currently FFHQ
  alone biases toward studio-lit Western 25-35yo portraits.
- **Curriculum option** (deferred): if portrait quality stalls below
  some threshold, start from exp51's FFHQ-only checkpoint and
  fine-tune on the 3k mixed set. Might give exp50-on-FFHQ quality
  *and* exp50-on-legacy capability simultaneously.
- **In-the-wild face val split**: a third val with small / off-center /
  partially-occluded faces, real photos. Currently no val covers this
  honestly; legacy val is group photos but skewed toward peripheral
  subjects.
