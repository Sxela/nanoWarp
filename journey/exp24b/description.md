## exp24b — exp23 + scale=2.0 crops (resize 1024→512, random crop 256)

**Status: DONE 2026-05-11**

Hypothesis: scale=2.0 gives consistent stroke width (2x downscale from native)
with manageable crop variance (1/4 image area vs 1/16 for scale=4.0 or ~full
image for scale=1.10).

```bash
python3 experiments/010_img2img_photo2comics/train.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 --image-size 256 --batch-size 4 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 --lr-cosine \
    --grad-clip-norm 1.0 --no-source-encoder --source-dropout 0.0 \
    --method flow --flow-sigma-noise 0.05 --amp bf16 \
    --model-ch 88 --attn-resolutions "16,32,64" \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --aug-resize-scale 2.0 --aug-scale-jitter 0.0 \
    --sample-panel-steps 20 --checkpoint-every 5000 \
    --val-every 1000 --panel-every 1000 \
    --outdir out/exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k
```

Results: lpips_sq=0.168, lpips_vgg=0.304, ssim=0.642

**Worse than exp23** (scale=1.10). Intermediate scale (2x downscale) removes
fine stroke variation but crops still cover less structure per sample than
scale=1.10. No recovery spike seen (unlike scale=4.0), but quality ceiling is
lower. scale=1.10 (resize to ~281px, ~full image visible) remains best.

**Winner of exp23 vs exp24b**: exp23 → used as base for exp25.

---
