### exp22 — exp14v2 + `--lpips-aux-net vgg` (LPIPS backbone swap) — PLANNED

The other axis to test on the winning recipe: does LPIPS-VGG beat
LPIPS-squeeze when used as the *only* perceptual aux (no Gram, no
content-L1 on top)? This is a cleaner version of what exp13 was
originally supposed to do, but now against a known-good baseline.

Single-flag change from exp14v2: `--lpips-aux-net vgg` instead of squeeze.

```powershell
python scripts/train.py img2img-v1 data/photo2anime_1k `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --image-size 256 `
    --attn-resolutions "16,32,64" `
    --amp bf16 `
    --color-space srgb `
    --source-dropout 0.15 --lpips-weight 0.2 --lpips-aux-net vgg `
    --aug-resize-scale 1.5 --aug-scale-jitter 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --max-loss-spike-ratio 10.0 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp22_lpips_vgg_noenc_1k_256px_20k `
    --wandb-tags "flow,no-encoder,lpips-vgg,bf16,1k-dataset,256px,exp22" `
    --outdir out/exp22_lpips_vgg_noenc_1k_256px_20k
```

Notes:
- We've now confirmed lpips_vgg is the honest metric (not in any training
  loop here either, since we'd train against LPIPS-VGG features but the
  metric uses different layers/weights).
- Adds ~10ms/step (~3 min extra over 20k).
- LPIPS-squeeze metric will likely worsen because we removed it from the
  loss; lpips_vgg metric is the one that matters for this comparison.
