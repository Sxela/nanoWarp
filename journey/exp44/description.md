## exp44 — 100k progressive + mid aug + exp35 arch + LPIPS anneal

**Status: DONE 2026-05-16** (data ceiling reached at 1k pairs)

Combined the best-known levers into one long run:
- 100k progressive 128→256→512 (exp32 schedule, ~75k at 512)
- Mid-strength aug (exp33c envelope: scale 1.5, blur ≤ 2.0, jpeg ≥ 40)
- exp35 arch (decoder attn + source pyramid + FiLM)
- LPIPS cosine-anneal 0.2 → 0.0

Final val (@ 256, 25 batches, EMA, sample_steps=20):

| metric | exp44 | exp35 (20k) |
|---|---|---|
| lpips_sq | 0.123 | 0.124 |
| lpips_vgg | 0.238 | 0.240 |
| ssim | 0.685 | 0.689 |
| face_lpips_sq | 0.154 | **0.153** |
| face_lpips_vgg | 0.288 | **0.286** |
| face_ssim | 0.726 | **0.728** |

**5× compute on the same 1k dataset got essentially exp35's result.** The
architecture/training-recipe axis is saturated at 1k pairs — face metrics
all within noise floor of exp35.

**In-loop val/lpips_sq curve shows a U-shape**: bottoms around step 60k
(lpips_weight ≈ 0.07) at val/lpips_sq=0.140, then climbs back to ~0.18 as
the LPIPS weight approaches 0. `model_best.pt` is from near the bottom of
the U — that's the right checkpoint to deploy from this run, not the
final one. The 25-batch full-val measures higher than the in-loop val
because val_batches=8 during training has higher variance.

Implication: **annealing LPIPS all the way to 0 is too aggressive**. A
floor at ~0.05–0.10 should retain the perceptual force that prevents
late-training MSE blur. → exp45.

---
