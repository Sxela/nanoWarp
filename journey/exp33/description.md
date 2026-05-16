## exp33 — exp23 recipe (20k @ 256px bs=4) with the full exp32 aug stack

**Status: RUNNING 2026-05-14**

Clean A/B vs exp23 (lpips_sq=0.127) to isolate the aug stack's impact at
fixed compute. Single architectural delta vs exp23: none — only the
augmentation pipeline changes. exp32 confounds aug × progressive-res × 100k
steps; exp33 strips out the first two confounds.

**Aug stack** (vs exp23's `scale=1.10` + hflip):
- Shared geometry: zoom scale U[1.0, 2.5], rotate ±25°, perspective 0.15
  @ p=0.5, hflip p=0.5.
- Source-only color jitter: brightness/contrast/saturation ±0.3.
- Source-only degradation (clean_prob=0.2): resize-down+up p=0.3
  (factor U[0.25, 0.75]), Gaussian blur σ U[0.5, 3.0] p=0.7,
  JPEG quality U[30, 95] p=0.7.

**Recipe** (everything else matches exp23): mc=88, attn 16/32/64, no source
encoder (source-in-stem), flow FM, LPIPS-VGG weight 0.2, bf16,
lr 2e-4 → 1e-5 cosine, warmup 500, 20k steps.

```bash
WANDB_API_KEY=... bash scripts/run_exp33_aug32stack_at_exp23_recipe.sh
```

Script: `scripts/run_exp33_aug32stack_at_exp23_recipe.sh`
Outdir: `out/exp33_aug32stack_noenc_attn163264_bf16_mc88_256px_20k`

Running on Colab (~20 min wall-clock for 20k steps at 256px bs=4 on the
provisioned GPU — single iteration loop is fast enough that from-scratch is
the default for the follow-up experiments).

Results: TBD.

**Aug-scale risk**: exp24 crashed at `scale=4.0` (1/16 image area per crop →
hard-region recovery dip at step ~4k). exp33 uses U[1.0, 2.5] which is in
the regime that exp24 fell over in; watch the val curve at 4-6k steps for a
similar pattern. If present, follow-up with `--aug-scale-max 1.5`.

---
