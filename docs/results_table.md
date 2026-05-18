# nanoWarp single-frame results (fast reference)

All metrics: final `validate.py` @ 256 unless noted, 25 batches, EMA,
sample_steps=20. Face metrics: cv2 Haar cascade on val sources, crops
resized to 128 for LPIPS/SSIM. Δ = corruption-val gap (smaller = more
robust to JPEG/blur/resize-degraded sources).

Bold = best in column among 20k-step single-phase runs on legacy val.

## Single-frame, legacy val (100 group photos, original 1k-dataset split)

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
| **exp50** (20k) | **3k mixed** | minimal | exp35 | 51M | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Single-frame, val_portraits (200 FFHQ portraits — meaningful face signal)

Established 2026-05-16. Old runs validated retroactively on this split.

| run | data | arch | face_lpips_sq | face_lpips_vgg | face_ssim | whole lpips_sq | whole ssim | Δ lpips_vgg |
|---|---|---|---|---|---|---|---|---|
| exp25 (80k) | 1k synth | base | 0.169 | 0.345 | 0.500 | 0.216 | 0.392 | — |
| exp35 (20k) | 1k synth | +dec_attn+pyramid | 0.178 | 0.370 | 0.477 | 0.215 | 0.384 | — |
| exp50 (20k) | 3k mixed | exp35 arch | 0.124 | 0.285 | 0.544 | 0.170 | 0.444 | +0.037 |
| exp51 (20k) | 2.3k FFHQ-only | exp35 arch | 0.122 | 0.280 | 0.550 | 0.168 | 0.448 | **+0.031** |
| **exp52 (80k)** | **3k mixed** | exp35 arch | **0.101** | **0.244** | **0.579** | **0.145** | **0.459** | +0.045 |
| exp53 (20k) | 3k mixed | exp35 arch + LANCZOS resize | TBD | TBD | TBD | TBD | TBD | TBD |

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
