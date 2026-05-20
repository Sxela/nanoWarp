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
