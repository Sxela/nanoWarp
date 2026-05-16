### exp13 — exp10 + VGG-backbone LPIPS aux — PLANNED

Single-flag follow-up. Switches the LPIPS auxiliary loss backbone from
SqueezeNet (~700k params) to VGG16 (~14M params).

Motivation:
- All three architectures tested at 128px (exp08-lpips, exp10, exp11) hit
  the same LPIPS ceiling around 0.152-0.153. One plausible cause: the
  SqueezeNet aux loss is too crude to drive the model past a certain
  perceptual quality floor.
- The classic style-transfer literature (Gatys, Johnson, AnimeGAN) all uses
  VGG features specifically because deeper VGG layers (`relu4_*`, `relu5_*`)
  encode texture/style signal well — which is exactly what anime stylization
  is about.
- LPIPS-VGG retains LPIPS's learned reweighting from the BAPPS dataset on
  top of the larger VGG backbone.

Cost: ~10 ms/step extra at bs=4/128px (vs ~2 ms for SqueezeNet). ~3 min
extra total wall-clock over 20k steps. Negligible.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.2 --lpips-aux-net vgg `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp13_vgg_lpips_noenc_attn832_bf16_mc88_20k `
    --wandb-tags "flow,no-encoder,lpips-vgg,attn-multiscale,bf16,exp13" `
    --outdir out/exp13_vgg_lpips_noenc_attn832_bf16_mc88_20k
```

Note: the **validation metric stays on LPIPS-SqueezeNet** for continuity
with exp01-exp11 numbers. exp13 results will be directly comparable to
exp10. If exp13 improves on the same SqueezeNet-LPIPS metric, the bigger
VGG aux loss was the bottleneck.

Predictions:
- If LPIPS-squeeze improves > 3% vs exp10 → VGG features were a real
  bottleneck for our perceptual ceiling. Adopt VGG as the new default
  aux loss for the 1k-pair dataset run too.
- If LPIPS-squeeze ≈ exp10 → the ceiling is data-bound, not aux-loss-bound.
  Stay with SqueezeNet (cheaper) and focus on data.
