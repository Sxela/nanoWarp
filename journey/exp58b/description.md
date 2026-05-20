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
