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
