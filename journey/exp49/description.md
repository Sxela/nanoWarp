## exp49 — UNet + 1k @ 128 bootstrap + 19k @ 256, exp35 arch + clean aug

**Status: DONE 2026-05-16** (LPIPS tied, SSIM regressed — 128 bootstrap didn't help)

Direct A/B vs exp35: same compute (5.24B pixel-samples), only
difference is whether the first 5% of training is at 128px (bs=16) or
256px (bs=4). Answers "does a 128px bootstrap help final 256 quality?"

Final val on legacy val/:

| metric | exp35 | exp49 (+128 bootstrap) |
|---|---|---|
| lpips_sq | **0.124** | 0.129 (tied) |
| lpips_vgg | **0.240** | 0.255 |
| ssim | **0.689** | 0.530 (big regression) |
| face_ssim | **0.728** | 0.619 |

LPIPS metrics essentially tied, but SSIM cratered. Unusual divergence —
outputs are *perceptually* similar but *pixel-wise* less aligned. The
1k phase change at 128 disrupted optimizer / EMA state in a way 19k
follow-up steps didn't fully recover.

**Answer to "does 128 bootstrap help"**: no, at 1k pairs. Compressed
progressive is dead at this scale. exp35's single-shot 256 stays
canonical.

---

## Temporal / video experiments

All temporal experiments (exp27 onwards) are documented in
[captains_log_video.md](captains_log_video.md).

---

## dual-LPIPS metric (2026-05-11)

validate.py now reports both `mean_lpips_squeeze_sampled` (continuity with
exp01–15) and `mean_lpips_vgg_sampled` (out-of-loop honest check). 28% gap
observed on exp08-noenc (0.163 sq vs 0.209 vgg). Always report both going forward.

---

## Corruption-robustness Δ-metric (2026-05-14)

Both the in-loop val ([train_exp32_prog512.py](../experiments/010_img2img_photo2comics/train_exp32_prog512.py))
and the final validation pass ([validate.py](../experiments/010_img2img_photo2comics/validate.py))
now run a second sampling pass per val batch against a deterministically
corrupted source, alongside the standard clean-source pass.

**Corruption** ([src/img2img/metrics.py::val_corrupt](../src/img2img/metrics.py)):
fixed mid-strength so numbers are comparable across runs/steps —
`resize 0.5× then bilinear-up → JPEG q=60 roundtrip → Gaussian blur σ=1.0`.
Roughly equivalent to a moderately-compressed web image. Strictly easier
than the worst-case training corruption (which goes up to σ=3 / q=30 /
resize 0.25× in exp32-style training).

**New metrics in `val_metrics.json` and wandb**:
- `mean_lpips_{squeeze,vgg}_corrupted` — same val data, corrupted source.
- `delta_lpips_{squeeze,vgg}` — corrupted minus clean. Smaller = more robust.

**Interpretation**: clean-trained models (exp23/exp25) show big Δ (∼0.10
lpips_vgg) because corruption is OOD for them. Corruption-trained models
(exp32+) show ∼0.05–0.07. The Δ distills "how much does this model fall
apart when the input isn't pristine" into one number, retroactively
backfillable by re-running validate.py on saved checkpoints.

**Reference numbers** (each model at its step 20k checkpoint, val at 256px,
EMA, 25 batches × bs=4, sample_steps=20):

| run | aug stack | clean lpips_vgg | corrupt lpips_vgg | **Δ lpips_vgg** | clean lpips_sq | corrupt lpips_sq | Δ lpips_sq |
|------|-----|------|------|------|------|------|------|
| exp25 (step 20k) | scale=1.10 + hflip | **0.234** | 0.350 | **+0.116** | 0.128 | 0.215 | +0.088 |
| exp32 (step 20k) | scale∈[1.0,2.5] + full corruption | 0.265 | **0.329** | **+0.064** | 0.142 | **0.185** | +0.043 |

Crossover: on corrupted source, exp32 wins (0.329 vs 0.350 lpips_vgg);
on clean source, exp25 wins (0.234 vs 0.265). The corruption-trained model
trades ~13% clean-quality for ~46% smaller robustness Δ and outright wins
when input isn't pristine — which is the regime real video frames live in.

**Going forward**: every `run_exp*.sh` end-of-run validate.py call now
emits both clean and corrupted numbers + Δ. The Δ should be reported
alongside lpips_vgg in any A/B summary; if Δ is large, the clean-val
number alone is misleading about real-world quality.

---

## Known bugs / lessons

- **2026-05-11 lesson: GAN weight 0.1 dominates over LPIPS 0.2.**
  At hinge equilibrium g_gan~1.5, contribution = 0.1×1.5 = 0.15 vs LPIPS
  contribution ~0.2×0.06 = 0.012. GAN is 12× larger than LPIPS in practice.
  For GAN as a weak regulariser, target gan_weight so that g_gan contribution
  ≈ LPIPS contribution, i.e. gan_weight ≈ 0.012/1.5 ≈ 0.008. 0.005 is in range.

- **2026-05-11 lesson: native-res crops (scale=4.0) cause 6k-step loss spike
  recovery.** 256px crops from 1024px images capture 1/16 image area — too
  sparse. Background-heavy crops dominate early training; face-heavy crops
  appear later and cause a loss jump. scale=2.0 (1/4 area) avoids the spike
  but still underperforms scale=1.10 (~full image). Confirmed ranking:
  scale=1.10 > scale=2.0 >> scale=4.0 on lpips_vgg (0.234 vs 0.304 vs 0.449).

- **2026-05-11 lesson: mc=176 hits capacity ceiling at 908 pairs/256px.**
  Both resize_conv and pixel_shuffle upsamplers showed periodic artifacts.
  Model converges to identity mapping. mc=88 (44M params) is the practical
  limit for this dataset scale.

## Open follow-ups

- ~~Compute "source-as-prediction" baseline SSIM/LPIPS.~~ **Done 2026-05-10.**
  Floor at 128px val: SSIM 0.617 / LPIPS 0.199. Floor at 256px val:
  SSIM 0.557 / LPIPS 0.299. Best 128px model: SSIM 0.740 / LPIPS 0.153.

- ~~Trainer `--resume` support.~~ **Done 2026-05-10.** Checkpoints
  (intermediate and final) include model + EMA + optimizer state + step.
  See exp08 resume command in its section.

- **Rename `FlowConfig.timesteps`** → `time_embedding_scale`. Currently
  `timesteps=1000` is misleading because FM has no discretized timestep
  schedule — `t` is continuous in [0,1] and we multiply by 999 only for
  the sinusoidal `TimeMLP`. Cleanup-only; touches checkpoint configs.

- **Best-checkpoint selection (val-LPIPS early stopping).** exp10/exp11
  showed LPIPS can regress past the optimum while SSIM/MSE keep improving.
  A trainer hook that re-validates every checkpoint and saves
  `best_lpips.pt` would automate finding the optimum. ~30 lines.

- Replace `save_loss_plot` with log-y + rolling mean. Lower priority now
  that wandb handles smoothing.

- Latent flow matching once the pixel-space pipeline is solid. Big lift
  (need an autoencoder), parked for when we have the bigger dataset
  results in hand.

- **Ablate `freeze=all` vs `freeze=partial`** on the encoder-on path now
  that we know freeze=all is stable. exp08-lpips uses freeze=all; checking
  whether freeze=partial would have worked too (and unlocked the deeper
  ResNet layers for task adaptation) would be informative.

- **Cheap SOTA-inspired patterns to consider** (post-exp14, see survey
  earlier in this document):
  - Zero-init gate on `FuseBlock` output (ControlNet trick): ~3 lines.
    Training stability.
  - Multi-scale L1/L2 loss alongside main loss (SR convention): ~10 lines.
    Small perceptual gain.
  - Charbonnier loss replacing pure MSE in the FM/diffusion path:
    trivial, minor numerical-stability benefit.
  - Window attention (SwinIR-style) at high resolutions: medium effort,
    relevant if we go ≥ 512px.
