## exp58 — logit-normal t-sampling (SD3/EDM-style)

**Status: DONE 2026-05-19** — finished anyway, **catastrophic
regression** on portraits (+44% face_lpips_sq), confirms endpoint
starvation theory cleanly.

Code change: added `t_sample_mode` / `t_sample_mu` / `t_sample_sigma`
fields to `FlowConfig`, branched in `flow.py:training_loss`. Default
remains `uniform` — exp50/52/56 reproduce.

Single-flag delta vs exp50: `--t-sample-mode logit_normal --t-sample-mu 0 --t-sample-sigma 1`.

Default flow training samples t ~ U[0,1]. Endpoints (t=0=source,
t=1=target) are "easy" — model just learns the full delta. Logit-normal
(t=sigmoid(N(mu,sigma))) peaks at 0.5 with mu=0, biasing training
toward the hard middle of the path where x_t is a mixed interpolant
and the model has to predict velocity from a partial signal. SD3 and
the Karras EDM family report consistent gains from this.

Smoke confirmed: 1000-sample empirical distribution has tighter std
(0.21 vs uniform's 0.28) and more mass in [0.4, 0.6] (29% vs 21%).

20k @ 256px bs=4 vs exp50. Script: `scripts/run_exp58_logit_normal_t_at_exp50_recipe.sh`

**Results vs exp50 (sigma=1.0, ran to completion)**:

| split | metric | exp50 | exp58 (sigma=1.0) | Δ |
|---|---|---|---|---|
| val_portraits | **face_lpips_sq** | **0.124** | **0.179** | **+44% LOSS** |
| val_portraits | face_lpips_vgg | 0.285 | 0.368 | +29% LOSS |
| val_portraits | face_ssim | 0.544 | 0.436 | -20% LOSS |
| val_portraits | whole lpips_sq | 0.170 | 0.210 | +24% LOSS |
| val_portraits | whole ssim | 0.444 | 0.386 | -13% LOSS |
| val_portraits | Δ_lpips_vgg | 0.037 | 0.041 | +11% (mild loss) |
| legacy val | face_lpips_sq | 0.201 | 0.223 | +11% LOSS |
| legacy val | face_ssim | 0.605 | 0.576 | -4.8% (tie/loss) |

Root cause confirmed empirically: at sigma=1.0, only **0.2%** of
training samples land at t<0.05 vs uniform's 5% — endpoints were **25×
starved**. Inference walks the ODE uniformly from t=0→1, so the first
Euler step `x = source + dt·v(·, t=0)` queried a model that effectively
never saw t≈0 during training. Trajectory corrupted from step 1 onward.

Surprise: empirical distribution math was much harsher than the
"peaked at 0.5" intuition. At sigma=1.0, only 1.4% of samples have
t<0.10 (vs uniform's 10%), and 31% land in [0.4, 0.6] (vs uniform's
20%) — that's a 7× concentration, not a "mild" bias. SD3 reports gains
with similar config but at 8B params + billions of samples; at our
~50M / 80k regime, the endpoints aren't optional.

Sweet-spot reanalysis across sigma values:

| sigma | t<0.05 | t<0.10 | [.4,.6] | t>0.95 | verdict |
|---|---|---|---|---|---|
| uniform | 5.0% | 10% | 20% | 5.0% | baseline |
| **1.00** (exp58) | **0.2%** | **1.4%** | **31%** | **0.2%** | endpoints 25× starved |
| **1.50** (exp58b) | 2.5% | 7.1% | 21% | 2.5% | sweet spot |
| 2.00 | 6.9% | 13.6% | 16% | 7.1% | bimodal — defeats purpose |

At sigma≥2.0 the underlying Gaussian is wide enough that sigmoid pushes
mass to BOTH tails — distribution is no longer mid-peaked.

---
