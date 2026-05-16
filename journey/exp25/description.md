## exp25 — exp23 recipe × 80k steps (long run)

**Status: DONE 2026-05-12**

Best recipe from ablation study (exp23: LPIPS-VGG, mc=88, attn 16/32/64, bf16,
scale=1.10, no encoder) extended to 80k steps to test continued improvement.

```bash
python3 experiments/010_img2img_photo2comics/train.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 80000 --image-size 256 --batch-size 4 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 --lr-cosine \
    --grad-clip-norm 1.0 --no-source-encoder --source-dropout 0.0 \
    --method flow --flow-sigma-noise 0.05 --amp bf16 \
    --model-ch 88 --attn-resolutions "16,32,64" \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --aug-resize-scale 1.10 --aug-scale-jitter 0.10 \
    --sample-panel-steps 20 --checkpoint-every 10000 \
    --val-every 5000 --panel-every 5000 \
    --wandb-run-name exp25_lpipsvgg_80k_from_exp23 \
    --outdir out/exp25_lpipsvgg_80k_from_exp23
```

Checkpoint progression (validated with 25 batches, 20 sample steps, EMA):

| step | lpips_sq | lpips_vgg | ssim  |
|------|----------|-----------|-------|
| 10k  | 0.133    | 0.243     | 0.671 |
| 20k  | 0.128    | 0.234     | 0.688 |
| 30k  | 0.124    | 0.228     | 0.698 |
| 40k  | 0.120    | 0.223     | 0.702 |
| 50k  | 0.117    | 0.219     | 0.708 |
| 60k  | 0.116    | 0.218     | 0.711 |
| 70k  | 0.116    | 0.218     | 0.712 |
| 80k  | 0.115    | 0.217     | 0.712 |

**Findings**: Monotonic improvement across all metrics. lpips_sq at 20k (0.128)
exactly matches exp23, confirming recipe reproducibility. Improvement rate
slows sharply after 60k: 10k→60k averages −0.005 lpips_sq per 10k steps;
60k→80k gains only −0.001 total. **Diminishing returns past 60k steps.**
Best checkpoint for deployment: 80k (final model.pt) — marginal gains justify
the full run but 60k is nearly as good. SSIM 0.712 vs 0.557 source floor.

---
