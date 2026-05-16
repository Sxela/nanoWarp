## exp36 — exp33 recipe + DiT bottleneck

**Status: WIRED 2026-05-14** (ready to launch after exp33/34/35 land)

Replaces the convolutional bottleneck (`mid_attn` + `mid2 ResBlock`) with a
stack of 4 DiT-XL-style transformer blocks operating on the flattened
(H/16 × W/16, cm=704) token grid. `mid1` (the c4 → cm channel-widener
ResBlock) is preserved upstream so the DiT stack always sees constant width.

**Block structure** (per DiT block):
- `LayerNorm` (elementwise_affine=False) → adaLN-zero modulation
  (shift_msa, scale_msa, gate_msa from a single Linear(t_emb_dim → 6·D)
  with zero-init weight and bias)
- MHSA (qkv proj + scaled_dot_product_attention + out proj)
- gated residual
- `LayerNorm` → adaLN-zero modulation (shift_mlp, scale_mlp, gate_mlp)
- MLP (Linear → GELU → Linear, hidden=4D)
- gated residual

Zero-init adaLN gates → block emits its input unchanged at step 0 →
no-DiT checkpoints load cleanly via strict=False, identical at-init forward.

**Positional embeddings**: 2D sinusoidal, size-agnostic so the same DiT
stack works at the 8×8 / 16×16 / 32×32 bottleneck grids that arise from
the 128 / 256 / 512px training phases.

**Heads**: auto-picked to keep `head_dim` a Flash-attention-friendly
power of 2. At cm=704: head_dim=64, num_heads=11.

**Param cost**: ~28M added (49M → ~77M total). Outside "same param budget"
territory — explicit choice to test maximum DiT capacity. Halve to ~14M
with `--num-dit-blocks 2` if exp36 is competitive on quality but the param
bloat is a problem for downstream temporal fine-tuning.

**Recipe**: identical to exp33 except `--use-dit-bottleneck` (+ optional
`--num-dit-blocks` and `--dit-mlp-ratio`).

```bash
WANDB_API_KEY=... bash scripts/run_exp36_dit_bottleneck_at_exp33_recipe.sh
```

Script: `scripts/run_exp36_dit_bottleneck_at_exp33_recipe.sh`
Outdir: `out/exp36_dit_bottleneck_noenc_attn163264_bf16_mc88_256px_20k`

Results: TBD.

**Why DiT over windowed attention at higher resolutions**: the windowed-attn
path (see [model_architecture.html](model_architecture.html) notes) would
add spatial-attention coverage at 128/256px levels that aren't currently
attended; DiT instead changes the *kind* of mixing at the bottleneck where
attention is already happening. The captain's log discussion concluded that
SD/SDXL conventionally skip full-res attention because the source-in-stem
path already provides full-res spatial information, so adding higher-res
attention is mostly buying high-frequency texture coherence rather than new
spatial reasoning. DiT at the bottleneck is the "smarter mixing at the
right resolution" direction instead.

---

## Results summary (clean-val @ 256px EMA, 25 batches, all archs auto-detected, 2026-05-15)

All single-frame runs at 20k steps unless noted. Best in each column **bold**.
Δ = corruption-val gap (smaller = more robust to degraded inputs).

| run | aug | arch + loss tweaks | params | lpips_sq | lpips_vgg | ssim | face_lpips_sq | face_lpips_vgg | face_ssim | Δ lpips_vgg |
|---|---|---|---|---|---|---|---|---|---|---|
| exp23 | minimal (scale=1.10) | base | 49M | 0.127 | **0.234** | 0.689 | — | — | — | — |
| exp25 (20k) | minimal | base | 49M | 0.128 | **0.234** | 0.688 | 0.157 | 0.289 | 0.728 | +0.116 |
| exp25 (80k) | minimal | base | 49M | **0.115** | 0.217 | 0.712 | — | — | — | — |
| exp32 (20k) | full corruption + scale 2.5 | base | 49M | 0.142 | 0.265 | 0.672 | 0.173 | 0.316 | 0.718 | +0.064 |
| exp32 (100k) @ 256 | full corruption | base | 49M | 0.178 | 0.321 | 0.638 | 0.209 | 0.364 | 0.698 | +0.058 |
| exp32 (100k) @ 512 | full corruption | base | 49M | 0.154 | 0.300 | 0.629 | 0.186 | 0.345 | 0.674 | **+0.040** |
| exp33 | full aug stack | base | 49M | 0.168 | 0.308 | 0.639 | — | — | — | — |
| exp33b | scale=1.5 + full corrupt | base | 49M | 0.148 | 0.274 | 0.659 | — | — | — | — |
| exp37 | minimal | + decoder attn | 51M | 0.126 | 0.242 | 0.684 | 0.156 | 0.289 | 0.724 | +0.133 |
| **exp35** | minimal | + dec_attn + pyramid | 51M | 0.124 | 0.240 | 0.689 | **0.153** | **0.286** | 0.728 | +0.133 |
| exp36 | minimal | + dec_attn + pyramid + DiT(4 blk) | **79M** | 0.123 | 0.238 | 0.685 | 0.154 | 0.288 | 0.726 | +0.130 |
| exp38 | minimal | exp35 + contrastive w=0.1 | 51M | 0.124 | 0.239 | 0.686 | 0.154 | 0.288 | 0.727 | +0.132 |
| exp39 | minimal | exp35 + contrastive w=0.3 | 51M | 0.124 | **0.238** | 0.687 | 0.155 | 0.288 | 0.726 | +0.126 |
| exp40 | minimal | exp35 + VGG Gram (w=5000) | 51M | 0.144 | 0.284 | 0.624 | 0.180 | 0.343 | 0.670 | +0.150 |
| exp41 cfg=1.0 | minimal | exp35 + source_dropout=0.1 | 51M | 0.128 | 0.244 | 0.683 | 0.158 | 0.293 | 0.721 | +0.133 |
| exp41 cfg=2.0 | minimal | (same ckpt, CFG inference) | 51M | 0.290 | 0.419 | 0.363 | 0.331 | 0.485 | 0.457 | +0.119 |
| exp42 | minimal | exp35 + LPIPS anneal 0.2→0 | 51M | 0.129 | 0.229 | **0.700** | 0.161 | 0.289 | **0.744** | +0.159 |

**Visual eye-test winner**: **exp35** (constant LPIPS=0.2, decoder attn + pyramid). exp42 has the best metrics but produces visibly blurrier outputs — the LPIPS anneal removed the perceptual push, so MSE converged to a pixel-aligned centroid (sharp on metrics, smooth on eyes).

**Robustness winner**: **exp32-100k @ 512** (Δ=0.040). 75k steps at 512 with full corruption aug.

**Methodology notes added 2026-05-14/15**: corruption-Δ metric, face-region metrics via OpenCV Haar cascade (43 faces detected at 256px over 100 val pairs; 110 at 512px), pinned `--panel-keys 000942,000943,000921` close-up face panels saved alongside legacy first-batch panels, post-hoc panel tool at `scripts/face_panels.py`. Stateful torchmetrics LPIPS accumulator bug fixed in flow.py (was causing 10× training slowdown over 20k steps).

**Architectural ceiling at 1k pairs**: decoder attn, pyramid, DiT each add at best 0.001–0.005 lpips_vgg over the previous best. The dataset is saturated for these mid-sized arch changes; the next 0.01+ improvements have to come from data scale, training length, or fundamentally different recipes (e.g. high-σ flow, see exp43).

---
