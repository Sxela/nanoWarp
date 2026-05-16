## exp46 — compressed progressive (1k @ 128 + 4k @ 256 + 15k @ 512, exp35 arch + clean aug)

**Status: DONE 2026-05-16** (worse than exp35 at 256, similar to exp32-100k at 512)

The proven exp32 progressive shape (compute-balanced phases) compressed
into a 20k budget. At 512 the model trained for only 15k steps vs
exp32-100k's 75k, so undersamples the largest phase.

Final val:

| metric | exp46 @ 256 | exp46 @ 512 | exp35 @ 256 (ref) | exp32-100k @ 512 (ref) |
|---|---|---|---|---|
| lpips_sq | 0.186 | 0.140 (in-loop) | **0.124** | 0.154 |
| ssim | 0.452 | ~0.62 (in-loop) | **0.689** | 0.629 |
| Δ lpips_vgg | +0.081 | — | +0.133 | **+0.040** |

The 256 numbers are misleading — the model spent 75% of training at 512,
so 256 inference is OOD. In-loop 512 val (lpips_sq 0.140) is competitive
with exp32-100k 0.154 — at *roughly* the same per-step compute as exp32
but with a 5× shorter total budget. Phase 3 didn't get enough steps to
fully converge at 512.

**Lesson**: progressive training needs *enough steps per phase*, not just
the right shape. Compressed-to-20k starves the largest phase. Either go
full 100k (exp32) or skip progressive entirely (exp35 baseline).

---
