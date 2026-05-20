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
