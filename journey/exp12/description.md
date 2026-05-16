### exp12 — original spec (kept for reference)

After exp10 / exp11 confirmed an architectural ceiling at 128px (LPIPS ~0.152
across three different architectures), the natural next axis is resolution
rather than more architecture. Higher resolution gives:

- More pixels = more LPIPS sensitivity (the metric should resolve smaller
  differences, exposing whether our models still have headroom).
- Larger receptive-field demands at the same fractional levels (more work
  for attention).
- A fairer test of how the architecture scales for the upcoming 1k-pair
  dataset (which is being generated at 1024px anyway, so we'll downscale).

Same architecture as exp10 (no encoder, mc=88, FM, LPIPS 0.2, dropout,
bf16, dataloader perf). One change: `--image-size 256` and matching
`--attn-resolutions "16,32,64"` to preserve fractional attention coverage.

| level | feat at 128px | feat at 256px | exp10 attn (8,16,32) | exp12 attn (16,32,64) |
|---|---:|---:|---|---|
| h1 | 128 | 256 | — | — |
| h2 | 64 | 128 | — | — |
| h3 | 32 | 64 | ✓ | ✓ |
| h4 | 16 | 32 | ✓ | ✓ |
| bottleneck (mid_attn) | 8 | 16 | ✓ always-on | ✓ always-on |

Cost estimates (extrapolated from 128px bench, no fresh measurement
because the GPU is busy generating the 1k dataset):

- Step time: ~120-160 ms (vs 31 ms at 128px).
- VRAM: ~5-6 GB peak (vs 1.3 GB at 128px). Comfortable on a 16 GB 4090.
- 20k steps wall-clock: ~45 min with dataloader + bf16.

Color space: keep `srgb` for now. Linear RGB was a wash at 128px; revisit
when EXR data lands.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp12_256px_noenc_attn16_32_64_bf16_mc88_20k `
    --wandb-tags "flow,no-encoder,lpips,attn-multiscale,bf16,256px,exp12" `
    --outdir out/exp12_256px_noenc_attn16_32_64_bf16_mc88_20k
```

If VRAM gets tight, drop `--batch-size` from 4 to 2 (default in trainer is 4).

Predictions:
- LPIPS at 256px should be **lower in absolute terms** because the metric
  has more spatial bandwidth to penalize errors. Our 128px LPIPS of 0.152
  doesn't translate directly — what matters is the relative gap to the
  256px floor, which we'd want to also compute.
- Visual quality should be noticeably better — anime-style flat regions
  and edges show up more crisply at 256.
- Memorisation risk: same 287 pairs, more pixel content per pair, so it's
  unclear if data scarcity dominates more or less at this resolution.
  Train-val gap diagnostic worth running again after this finishes.
