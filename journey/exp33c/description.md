## exp33c — exp33b recipe (scale 1.5) with corruption tail dialled to a realistic web-video envelope

**Status: WIRED 2026-05-14**

Hypothesis: exp33b's clean-val regression vs exp23 (lpips_vgg 0.274 vs 0.234)
is partly the rare extreme tail of the corruption distribution burning
training capacity on inputs that never occur in real footage. Tightening the
tail should recover clean-val toward exp23 while keeping most of the
robustness gain (smaller corruption-val Δ than exp25's +0.116).

**Aug recipe deltas vs exp33b**:
- `--degrade-resize-min 0.25 → 0.5` (when fired: 4× area max instead of 16×)
- `--degrade-resize-max 0.75 → 0.9` (most resize cases very mild)
- `--corrupt-blur-max 3.0 → 2.0` (cuts out-of-focus extreme)
- `--corrupt-jpeg-min 30 → 40` (skips the worst block artifacts)

Everything else identical to exp33b: scale ∈ [1.0, 1.5], rotate ±25°,
perspective 0.15, color jitter ±0.3, hflip, exp23-style architecture (no
decoder attn / pyramid / DiT changes).

```bash
WANDB_API_KEY=... bash scripts/run_exp33c_milder_corruption.sh
```

Script: `scripts/run_exp33c_milder_corruption.sh`
Outdir: `out/exp33c_milder_corruption_noenc_attn163264_bf16_mc88_256px_20k`

**Goal**: clean lpips_vgg closer to exp23's 0.234 than exp33b's 0.274, and
corruption-val Δ still meaningfully below exp25's 0.116. Sweet spot is
clean ≈ 0.25 with Δ ≈ 0.08 — buys most of the robustness for half the
clean-val cost.

Results: TBD.

---
