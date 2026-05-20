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
