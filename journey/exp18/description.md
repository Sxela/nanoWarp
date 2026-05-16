### exp18 — exp15 with 2× model capacity (mc=88 → 128) — PLANNED

Capacity scaling test. exp15 hit LPIPS 0.162 with the 1k dataset at
mc=88 (~44M params). The question: is that the data ceiling or the
architecture ceiling? Doubling model capacity isolates the test.

Single-variable change vs exp15: `--model-ch 128` instead of 88.

| | exp15 (mc=88) | **exp18 (mc=128)** |
|---|---:|---:|
| total params | 44.9M | **93.6M** (2.08×) |
| step time (256px, bs=4, bf16) | 255 ms | 332 ms (+30%) |
| peak VRAM | 10.0 GB | 11.3 GB |
| 20k wall-clock | ~85 min | ~110 min |

VRAM fits at bs=4 on a 16GB 4090, so no effective-batch confound — clean
single-variable test.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 128 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --feature-content-weight 1.0 --feature-style-weight 5000.0 `
    --feature-loss-layers "8,15,22" `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp18_capacity2x_mc128_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,feature-loss,2x-capacity,256px,exp18" `
    --outdir out/exp18_capacity2x_mc128_1k_256px_20k
```

Predictions:
- If exp18 LPIPS ≤ 0.155 → capacity was partially limiting. Worth going
  to 4× (exp19, mc=176, bs=2 with effective-batch confound).
- If exp18 ≈ 0.162 → architecture's not the bottleneck at this scale.
  Stop scaling capacity, look at data variety / loss design.
- Visual check is the better signal for the "shape simplicity" concern
  — doubled capacity might let the model encode more complex shapes
  even if the LPIPS metric doesn't move much.
