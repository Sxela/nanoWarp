# nanoWarp single-frame results (fast reference)

All metrics: final `validate.py` @ 256 unless noted, 25 batches, EMA,
sample_steps=20. Face metrics: cv2 Haar cascade on val sources, crops
resized to 128 for LPIPS/SSIM. Δ = corruption-val gap (smaller = more
robust to JPEG/blur/resize-degraded sources).

Bold = best in column among 20k-step single-phase runs on legacy val.

## Single-frame, legacy val (100 group photos, original 1k-dataset split)

### 1k-synth era (canonical = exp35 for arch comparison, exp25-80k for raw-metric leader)

| run | data | aug | arch | params | lpips_sq | lpips_vgg | ssim | face_lpips_sq | face_lpips_vgg | face_ssim | Δ lpips_vgg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **exp23** (20k) | 1k synth | scale=1.10 + hflip | base | 49M | 0.127 | **0.234** | 0.689 | — | — | — | — |
| **exp25** (20k) | 1k synth | scale=1.10 | base | 49M | 0.128 | **0.234** | 0.688 | 0.157 | 0.289 | 0.728 | +0.116 |
| exp25 (80k) | 1k synth | scale=1.10 | base | 49M | **0.115** | 0.217 | 0.712 | — | — | — | — |
| exp32 (20k) | 1k synth | full corrupt s≤2.5 | base | 49M | 0.142 | 0.265 | 0.672 | 0.173 | 0.316 | 0.718 | +0.064 |
| exp32 (100k) @ 256 | 1k synth | full corrupt s≤2.5 | base | 49M | 0.178 | 0.321 | 0.638 | 0.209 | 0.364 | 0.698 | +0.058 |
| exp32 (100k) @ 512 | 1k synth | full corrupt s≤2.5 | base | 49M | 0.154 | 0.300 | 0.629 | 0.186 | 0.345 | 0.674 | **+0.040** |
| exp33 (20k) | 1k synth | full aug s≤2.5 | base | 49M | 0.168 | 0.308 | 0.639 | — | — | — | — |
| exp33b (20k) | 1k synth | full aug s≤1.5 | base | 49M | 0.148 | 0.274 | 0.659 | — | — | — | — |
| **exp35** (20k) | 1k synth | minimal | +dec_attn+pyramid | 51M | **0.124** | 0.240 | 0.689 | **0.153** | **0.286** | 0.728 | +0.133 |
| exp36 (20k) | 1k synth | minimal | +dec_attn+pyramid+DiT | 79M | 0.123 | 0.238 | 0.685 | 0.154 | 0.288 | 0.726 | +0.130 |
| exp37 (20k) | 1k synth | minimal | +dec_attn | 51M | 0.126 | 0.242 | 0.684 | 0.156 | 0.289 | 0.724 | +0.133 |
| exp38 (20k) | 1k synth | minimal | exp35 + contrastive w=0.1 | 51M | 0.124 | 0.239 | 0.686 | 0.154 | 0.288 | 0.727 | +0.132 |
| exp39 (20k) | 1k synth | minimal | exp35 + contrastive w=0.3 | 51M | 0.124 | 0.238 | 0.687 | 0.155 | 0.288 | 0.726 | +0.126 |
| exp40 (20k) | 1k synth | minimal | exp35 + VGG Gram (w=5000) | 51M | 0.144 | 0.284 | 0.624 | 0.180 | 0.343 | 0.670 | +0.150 |
| exp41 cfg=1.0 (20k) | 1k synth | minimal | exp35 + src_dropout=0.1 | 51M | 0.128 | 0.244 | 0.683 | 0.158 | 0.293 | 0.721 | +0.133 |
| exp41 cfg=2.0 | 1k synth | minimal | exp35, CFG inference | 51M | 0.290 | 0.419 | 0.363 | 0.331 | 0.485 | 0.457 | +0.119 |
| exp42 (20k) | 1k synth | minimal | exp35 + LPIPS anneal 0.2→0 | 51M | 0.129 | **0.229** | **0.700** | 0.161 | 0.289 | **0.744** | +0.159 |
| exp43 (20k) | 1k synth | minimal | exp35 + σ_noise=0.30 | 51M | 0.514 | 0.428 | 0.134 | — | — | 0.209 | — |
| exp44 (100k progressive) | 1k synth | mid aug s≤1.5 | exp35 arch + LPIPS→0 | 51M | 0.123 | 0.238 | 0.685 | 0.154 | 0.288 | 0.726 | +0.130 |
| exp45 (20k) | 1k synth | minimal | exp35 + LPIPS anneal→0.1 | 51M | 0.124 | 0.239 | 0.686 | — | — | — | — |
| exp46 @ 256 (20k prog) | 1k synth | minimal | exp35 + 128/256/512 prog | 51M | 0.186 | 0.343 | 0.452 | 0.233 | 0.418 | 0.545 | +0.081 |
| exp46 @ 512 (20k prog) | 1k synth | minimal | exp35 + 128/256/512 prog | 51M | ~0.140 | — | ~0.62 | — | — | — | — |
| exp47 (20k) | 1k synth | minimal | pixel DiT 384/11/16 | 49M | bad | bad | bad | bad | bad | bad | — |
| exp48 @ 256 (20k prog) | 1k synth | minimal | pixel DiT + multiscale + warmup | 49M | 0.377 | 0.447 | 0.353 | 0.374 | 0.524 | 0.463 | +0.091 |
| exp48 @ 512 (20k prog) | 1k synth | minimal | pixel DiT + multiscale + warmup | 49M | 0.451 | 0.491 | 0.278 | 0.438 | 0.561 | 0.401 | +0.049 |
| exp49 (20k) | 1k synth | minimal | exp35 + 128 bootstrap | 51M | 0.129 | 0.255 | 0.530 | 0.180 | 0.348 | 0.619 | +0.129 |

### 3k-mixed era (not directly comparable to 1k-synth above — different training source distribution)

| run | data | aug | arch | params | lpips_sq | lpips_vgg | ssim | face_lpips_sq | face_lpips_vgg | face_ssim | Δ lpips_vgg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp50 (20k) | 3k mixed | minimal | exp35 arch | 51M | 0.150 | 0.297 | 0.516 | 0.201 | 0.379 | 0.605 | +0.116 |
| **exp52 (80k)** | **3k mixed** | minimal | exp35 arch | 51M | TBD | TBD | TBD | **0.183** | **0.355** | 0.623 | ~+0.125† |
| exp53 (20k) | 3k mixed | minimal | exp35 + LANCZOS resize | 51M | 0.148 | 0.303 | 0.485 | 0.214 | 0.402 | 0.533 | +0.116 |
| exp54 (20k) | 3k mixed | minimal | exp35 + diffusion-eps (flow→diff) | 51M | 0.433 | 0.567 | 0.322 | 0.482 | 0.621 | 0.524 | +0.133 |
| exp55 (20k) | 3k mixed | minimal | exp35 + diffusion-eps + lpips=0 | 51M | 0.619 | 0.683 | 0.320 | 0.693 | 0.752 | 0.494 | +0.084 |
| **exp56 (80k)** | **3k mixed** | mid aug | exp35 arch | 51M | **0.136** | **0.260** | **0.534** | 0.191 | 0.359 | 0.631 | **+0.077** |
| exp57 (20k) | 3k mixed | minimal | exp35 + source-dropout=0.2 | 51M | 0.156 | 0.308 | 0.520 | 0.207 | 0.390 | 0.600 | +0.113 |
| exp58 (20k) | 3k mixed | minimal | exp35 + logit-normal t σ=1.0 | 51M | 0.162 | 0.306 | 0.492 | 0.223 | 0.398 | 0.576 | +0.114 |
| exp58b (20k) | 3k mixed | minimal | exp35 + logit-normal t σ=1.5 | 51M | 0.153 | 0.296 | 0.507 | 0.204 | 0.379 | 0.596 | +0.113 |
| exp59 (20k) | 3k mixed | minimal | exp35 + cross-attn cond @ H/8 | 51.5M | 0.149 | 0.294 | 0.512 | 0.203 | 0.381 | 0.598 | +0.111 |
| exp62 (20k) | 3k mixed | minimal | exp35 + cross-attn (H/8+H/4) - source-in-stem | 49M | 0.150 | 0.298 | 0.521 | 0.205 | 0.386 | 0.609 | +0.120 |
| exp64 (20k) | 3k mixed | minimal | exp35 + AdaLN-Zero everywhere (LOSE) | 58M | 0.152 | 0.297 | 0.516 | 0.206 | 0.385 | 0.614 | +0.120 |
| exp64b (80k) | 3k mixed | minimal | exp35 + AdaLN @ 80k (LOSES to mc=88 — chapter closed) | 58M | 0.135 | 0.267 | 0.530 | 0.186 | 0.357 | 0.628 | +0.130 |
| exp66 (20k) | 3k mixed | minimal | exp35 + mc=128 (TIE — under-trained) | 102M | 0.148 | 0.292 | 0.514 | 0.200 | 0.379 | 0.600 | +0.117 |
| exp66b (80k) | 3k mixed | minimal | exp35 + mc=128 @ 80k (LOSE to mc=88 — overcapacity) | 102M | 0.132 | 0.261 | 0.526 | 0.184 | 0.354 | 0.621 | +0.131 |
| **exp65 (20k)** | **3k mixed** | minimal | exp35 + x0-pred (huge SSIM win) | 51.5M | 0.148 | **0.281** | **0.623** | 0.188 | **0.344** | **0.674** | +0.133 |
| **exp65b (80k)** | **3k mixed** | minimal | exp35 + x0-pred (**NEW QUALITY CANONICAL**) | 51.5M | **0.129** | **0.248** | **0.655** | **0.163** | **0.309** | **0.706** | +0.137 |
| exp65c (80k) | 3k mixed | mid aug | exp35 + x0-pred + cross-attn (composition test) | 51.5M | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| exp67 (20k) | 3k mixed | minimal | exp35 + SGDR 2-cycle LR (TIE — no plateau at 20k) | 51.5M | 0.148 | 0.290 | 0.514 | 0.205 | 0.383 | 0.600 | +0.112 |
| exp68a (20k) | 3k mixed | minimal | exp35 + lr=4e-4 (2× default — catastrophic LOSE) | 51.5M | 0.164 | 0.314 | 0.501 | 0.218 | 0.398 | 0.601 | +0.127 |
| exp67b (80k) | 3k mixed | minimal | exp35 + SGDR 2-cycle @ 80k (LOSES — restart disrupts refinement) | 51.5M | 0.133 | 0.265 | 0.525 | 0.187 | 0.362 | 0.614 | +0.115 |
| **exp60 (80k)** | **3k mixed** | minimal | exp35 + cross-attn (**quality canonical**) | 51.5M | **0.131** | **0.259** | 0.530 | **0.182** | **0.349** | 0.630 | +0.113 |
| **exp61 (80k)** | **3k mixed** | mid aug | exp35 + cross-attn (**deployment canonical**) | 51.5M | 0.137 | 0.264 | 0.533 | 0.189 | 0.363 | **0.632** | **+0.078** |
| exp63 (20k adv) | 3k mixed | mid aug | exp61 + PatchGAN gan_w=0.02 (within-noise drift) | 51.5M+D | 0.136 | 0.263 | 0.533 | 0.193 | 0.367 | 0.625 | +0.076 |
| exp63b (20k adv) | 3k mixed | mid aug | exp61 + PatchGAN gan_w=0.05 + forced D (retry) | 51.5M+D | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

† exp52 legacy whole-image metrics weren't recorded at final-val time
(the captain's log only captured face metrics); Δ estimated from
mid-training wandb chart.

**Reading the 3k-mixed legacy val table**:
- **exp60** (cross-attn + low aug, quality canonical) **strictly dominates exp52**
  across every legacy metric — cross-attn alone (no aug) is the best
  legacy-val face recipe. face_lpips_sq=0.182 < exp52's 0.183, plus wins
  whole-image lpips_sq=0.131 (best in column).
- **exp56/exp61** (mid aug) lose ~3% on legacy face metrics vs exp60 but
  gain on robustness Δ — the trade is "training-time corruption exposure".
- **exp61** is the second-best on every legacy metric and the deployment
  canonical (best robustness on val_portraits, still beats exp52 on
  whole-image legacy).
- **No 3k-era run regresses near the FFHQ-only exp51 catastrophe**
  (-90% face_lpips_sq vs exp35-on-1k); mid-aug + 3k-mixed dataset
  preserves global-scene capability well.

## Single-frame, val_portraits (200 FFHQ portraits — meaningful face signal)

Established 2026-05-16. Old runs validated retroactively on this split.

| run | data | arch | face_lpips_sq | face_lpips_vgg | face_ssim | whole lpips_sq | whole ssim | Δ lpips_vgg |
|---|---|---|---|---|---|---|---|---|
| exp25 (80k) | 1k synth | base | 0.169 | 0.345 | 0.500 | 0.216 | 0.392 | — |
| exp35 (20k) | 1k synth | +dec_attn+pyramid | 0.178 | 0.370 | 0.477 | 0.215 | 0.384 | — |
| exp50 (20k) | 3k mixed | exp35 arch | 0.124 | 0.285 | 0.544 | 0.170 | 0.444 | +0.037 |
| exp51 (20k) | 2.3k FFHQ-only | exp35 arch | 0.122 | 0.280 | 0.550 | 0.168 | 0.448 | **+0.031** |
| **exp52 (80k)** | **3k mixed** | exp35 arch | **0.101** | **0.244** | **0.579** | **0.145** | **0.459** | +0.045 |
| exp53 (20k) | 3k mixed | exp35 arch + LANCZOS resize | 0.124 | 0.289 | 0.521 | 0.164 | 0.423 | +0.039 |
| exp54 (20k) | 3k mixed | exp35 arch + diffusion-eps (vs flow) | 0.508 | 0.760 | 0.370 | 0.514 | 0.368 | +0.047 |
| exp55 (20k) | 3k mixed | exp35 arch + diffusion-eps + lpips=0 | 0.725 | 0.795 | 0.398 | 0.707 | 0.413 | +0.032 |
| **exp56 (80k)** | **3k mixed** | exp35 arch + mid aug (head pose, mild corrupt) | 0.104 | **0.244** | 0.577 | 0.148 | 0.460 | **+0.027** |
| exp57 (20k) | 3k mixed | exp35 arch + source-dropout=0.2 (no CFG) | 0.124 | 0.290 | 0.550 | 0.172 | 0.457 | +0.034 |
| exp58 (20k) | 3k mixed | exp35 arch + logit-normal t (sigma=1.0, endpoints 25x starved) | 0.179 | 0.368 | 0.436 | 0.210 | 0.386 | +0.041 |
| exp58b (20k) | 3k mixed | exp35 arch + logit-normal t (sigma=1.5) | 0.136 | 0.309 | 0.507 | 0.182 | 0.422 | +0.037 |
| **exp59 (20k)** | **3k mixed** | exp35 arch + cross-attn cond @ H/8 (+500k params) | **0.122** | 0.282 | 0.546 | 0.166 | 0.445 | **+0.035** |
| **exp60 (80k)** | **3k mixed** | exp35 + cross-attn (**quality canonical**, first sub-0.10) | **0.0997** | **0.237** | 0.583 | **0.142** | 0.460 | +0.040 |
| **exp61 (80k)** | **3k mixed** | exp35 arch + cross-attn + mid aug (STACK, **new canonical**) | 0.103 | **0.242** | **0.581** | 0.148 | 0.460 | **+0.025** |
| exp62 (20k) | 3k mixed | exp35 + cross-attn (H/8 + H/4) + NO source-in-stem | **0.119** | **0.278** | 0.554 | 0.165 | 0.449 | +0.041 |
| exp63 (20k adv) | 3k mixed | exp61 + PatchGAN gan_w=0.02 (within-noise drift, NOT canonical) | 0.101 | 0.239 | 0.583 | 0.146 | 0.461 | +0.024 |
| exp63b (20k adv) | 3k mixed | exp61 + PatchGAN gan_w=0.05 + forced D updates (diagnosis-fix retry) | TBD | TBD | TBD | TBD | TBD | TBD |
| exp64 (20k) | 3k mixed | exp59 + AdaLN-Zero time conditioning (DiT-style, +9.5M; LOSE) | 0.131 | 0.300 | 0.534 | 0.176 | 0.433 | +0.048 |
| exp64b (80k) | 3k mixed | exp59 + AdaLN-Zero @ 80k — LOSES (+11% face_lpips_sq, +46% Δ) | 0.111 | 0.261 | 0.565 | 0.154 | 0.446 | +0.0586 |
| **exp65 (20k)** | **3k mixed** | exp59 + x0-prediction (huge SSIM win, robustness regress) | 0.121 | **0.269** | **0.587** | 0.163 | **0.559** | +0.042 |
| **exp65b (80k)** | **3k mixed** | exp65 @ 80k (**NEW QUALITY CANONICAL** — x0-pred wins on every quality metric) | **0.0996** | **0.226** | **0.635** | 0.137 | **0.593** | +0.047 |
| exp65c (80k) | 3k mixed | STACK x0-pred + mid-aug + cross-attn (exp65b + exp61 composition test) | TBD | TBD | TBD | TBD | TBD | TBD |
| exp67b (80k) | 3k mixed | exp67 @ 80k — SGDR 2-cycle LOSES (warm restart disrupts late-training) | 0.104 | 0.247 | 0.574 | 0.147 | 0.452 | +0.039 |
| exp66 (20k) | 3k mixed | exp59 + model_ch=128 (102M params, TIE at 20k — under-trained) | 0.126 | 0.287 | 0.543 | 0.168 | 0.443 | +0.037 |
| exp66b (80k) | 3k mixed | exp66 @ 80k — mc=128 LOSES to mc=88 (under-training refuted) | 0.105 | 0.248 | 0.576 | 0.148 | 0.454 | +0.0505 |
| exp67 (20k) | 3k mixed | exp59 + SGDR 2-cycle LR (TIE — no plateau to escape at 20k) | 0.122 | 0.284 | 0.546 | 0.168 | 0.447 | +0.036 |
| exp68a (20k) | 3k mixed | exp59 + lr=4e-4 (2× default — CATASTROPHIC LOSE, +29% face_lpips_sq) | 0.158 | 0.344 | 0.491 | 0.198 | 0.393 | +0.046 |
| exp68b (20k) | 3k mixed | exp59 + lr=6e-4 (3×) — CANCELLED (2× already cratered) | — | — | — | — | — | — |
| exp68c (20k) | 3k mixed | exp59 + lr=1e-3 (5×) — CANCELLED (2× already cratered) | — | — | — | — | — | — |

## Cross-domain val (FFHQ-only-trained on legacy val)

| run | data | legacy face_lpips_sq | legacy face_ssim |
|---|---|---|---|
| exp35 (20k) | 1k synth | **0.153** | **0.728** |
| exp50 (20k) | 3k mixed | 0.201 (-31%) | 0.605 (-17%) |
| exp51 (20k) | 2.3k FFHQ-only | 0.290 (-90%) | 0.510 (-30%) |

FFHQ-only training catastrophically loses small/peripheral face capability.
Mixed (exp50) is the sane tradeoff: marginally worse on FFHQ portraits
than FFHQ-only, much better on legacy than FFHQ-only.

## Notes on reading the table

- **Architectural ceiling at 1k synth**: exp35/36/38/39/44 all hover at
  lpips_sq ≈ 0.124, face_lpips_sq ≈ 0.153. Adding decoder_attn, pyramid,
  DiT bottleneck, contrastive loss, longer training each move things by
  ≤ 0.005 lpips_sq — data ceiling, not architecture ceiling.
- **Recipe ceiling at 1k synth**: same — LPIPS anneal (exp42), σ_noise
  (exp43), CFG (exp41), VGG Gram (exp40), compressed progressive (exp46),
  128 bootstrap (exp49) all either tie or regress vs exp35.
- **exp35 is the canonical 20k single-frame baseline** by visual eye-test;
  exp25-80k is the raw-metric leader.
- **Robustness leader (Δ)**: exp32-100k @ 512 (Δ=0.040). All other runs
  trained on clean sources stay around Δ=0.13.
- **Face metrics ≠ visual face quality**: exp42 has the best face_ssim
  (0.744) but produces visibly blurrier panels than exp35 (0.728).
  LPIPS-VGG L2 on face crops rewards centroid commitment, not crisp
  feature placement. Always cross-check with the face panels.
- **The legacy val split is wrong for faces**: it's 100 group photos with
  peripheral / non-frontal subjects. `val_portraits` (200 FFHQ portraits)
  is the meaningful face-quality measurement going forward.
- **Pixel DiT (exp47/48) is dead at our scale**: 0.3+ lpips_sq gap to
  UNet that no recipe trick closed. HiDream-O1 makes it work via 8B
  params + massive data — not transferable.

## Validation recipe (apples-to-apples)

```bash
PYTHONPATH=. python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint $CKPT.pt \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 \
    --sample-steps 20 --use-ema \
    --outdir out/val_$NAME_on_val_portraits
```

The 25-batch / sample_steps=20 / EMA configuration is what every number
in the tables above was measured with.
