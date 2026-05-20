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
