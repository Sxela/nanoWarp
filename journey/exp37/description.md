## exp37 — exp23-equivalent recipe (minimal aug) + symmetric decoder attn

**Status: WIRED 2026-05-14** (architecture A/B on a clean baseline + a third
anchor for the corruption-Δ metric)

Two goals in one run:
1. **Architecture-only A/B against exp23 (lpips_vgg=0.234)**. exp34 stacks
   decoder attn on the full exp32 aug stack — which exp33's result showed
   costs ~0.07 lpips_vgg on clean before any architecture changes — so the
   architecture delta is hard to read. exp37 stacks the same architecture
   change on the proven low-aug exp23 baseline so the read is clean.
2. **Corruption-Δ anchor for a clean-trained 20k checkpoint.** The original
   exp25 checkpoint that would have filled this anchor was lost before the
   Δ metric was wired. exp37 provides a fresh clean-trained 20k anchor to
   compare against exp32's Δ=+0.064 (corruption-trained anchor).

**Aug settings ≈ exp23** (exp32 script reproduces exp23-style behaviour when
geometric/color aug is dialled to ~identity and corruption is fully skipped):
- `--aug-scale-min 1.0 --aug-scale-max 1.2` (matches exp23's
  `resize_scale=1.10` + `scale_jitter=0.10`)
- `--aug-rotate-deg 0.0`, `--aug-perspective-prob 0.0`
- `--aug-brightness 0 --aug-contrast 0 --aug-saturation 0` (no color jitter)
- `--clean-prob 1.0` → degradation pipeline fully skipped
- hflip stays at p=0.5 (matches exp23, no CLI knob)

**Architecture delta**: identical to exp34 (`--use-decoder-attn` → adds
`attn_dec3` at H/4=64 and `attn_dec4` at H/8=32, mirroring encoder attn3/4
at the same resolutions).

```bash
WANDB_API_KEY=... bash scripts/run_exp37_decoder_attn_at_exp23_recipe.sh
```

Script: `scripts/run_exp37_decoder_attn_at_exp23_recipe.sh`
Outdir: `out/exp37_decoder_attn_at_exp23_recipe_noenc_attn163264_bf16_mc88_256px_20k`

**What to look for** in `out/val_exp37_final_256px/val_metrics.json`:
- `mean_lpips_vgg_sampled` vs exp23's **0.234** — the decoder-attn delta on clean.
  - < 0.234 → decoder attn helps; stack on future runs.
  - ≈ 0.234 → neutral at this dataset size; capacity isn't the bottleneck.
  - > 0.234 → hurts; not enough data to train the extra modules.
- `delta_lpips_vgg` (corruption-val gap) — third anchor on the Δ axis next to
  exp25 (Δ=+0.116, clean-trained, 20k) and exp32 (Δ=+0.064, corruption-trained,
  20k). Same baseline aug as exp25 plus the architecture change.

Results: TBD.

**Independent of exp37, exp33c is wired** as the robustness-recipe test
(see entry below). exp33c sweeps the aug-recipe axis; exp37 sweeps the
architecture axis. Both can run in either order.

---
