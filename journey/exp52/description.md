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
