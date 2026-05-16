### exp19 — exp14v2 + `--lpips-weight 0.4` (double LPIPS aux) — PLANNED

After the fair-step revalidation showed exp14v2 (LPIPS only) is the
strongest recipe, the natural follow-up is to lean harder on the signal
that's working. Single-flag change: double the LPIPS aux weight from 0.2
to 0.4.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.4 `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp19_lpips04_noenc_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips,lpips-0.4,bf16,1k-dataset,256px,exp19" `
    --outdir out/exp19_lpips04_noenc_1k_256px_20k
```

Predictions:
- If exp19 lpips_vgg < 0.248 (exp14v2 at 20k) → LPIPS weight was
  under-tuned. Adopt 0.4 as new default.
- If exp19 ≈ exp14v2 → LPIPS at 0.2 was already saturating. Stay there.
- If exp19 worse → 0.4 over-emphasizes perceptual at cost of MSE/structure.
  Bracket the optimum: try 0.3 next.
