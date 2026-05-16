### exp15 — exp10 + VGG feature loss (content L1 + Gram-matrix style L1) — PLANNED

Adds fastai-style `FeatureLoss` (Gatys / Johnson texture-aware loss) on
top of the exp10 architecture. Where LPIPS measures "do these two images
look similar?", Gram-matrix style loss measures "do these two images
have the same texture statistics?" — which is what stylization actually
demands. Anime stylization is fundamentally a texture transformation
(photo skin/hair textures → anime flat regions + ink lines), so this is
a plausibly high-leverage axis.

[src/img2img/feature_loss.py](../src/img2img/feature_loss.py) implements:
- Frozen VGG16 forward at chosen layers (default: 8, 15, 22 = after
  relu2_2 / relu3_3 / relu4_3, the fastai/Johnson convention).
- **Content L1**: feature distance at each layer.
- **Style L1**: Gram-matrix distance at each layer. Gram is the
  channel-wise covariance, capturing texture statistics independent of
  spatial position.
- ImageNet mean/std normalisation before VGG; input clamped to [0, 1].
- For `--color-space linear_rgb`, converts to sRGB at the boundary.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --no-source-encoder --model-ch 88 `
    --attn-resolutions "8,16,32" `
    --amp bf16 `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --feature-content-weight 1.0 --feature-style-weight 5000.0 `
    --feature-loss-layers "8,15,22" `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 5000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp15_featureloss_noenc_attn832_bf16_mc88_20k `
    --wandb-tags "flow,no-encoder,lpips,feature-loss,gram-style,bf16,exp15" `
    --outdir out/exp15_featureloss_noenc_attn832_bf16_mc88_20k
```

**Loss magnitude warning**: `style_weight=5000` (fastai/Johnson default)
makes the style term dominate the total loss. Smoke test at step 30
showed total loss ~10-30 vs diffusion ~0.02 and lpips ~0.05. **This is
expected** — the absolute total isn't directly comparable to other
exp totals. What matters is the per-term values logged separately
(`train/feature_content`, `train/feature_style`) and the **val LPIPS**
on the standard SqueezeNet metric.

Tuning candidates if results look off:
- Style dominating too much → drop to `--feature-style-weight 1000` or `500`.
- Texture matching too aggressive (output looks "over-stylized") → lower style weight.
- Texture not transferring (output stays photo-like) → raise style weight or add deeper VGG layers (e.g. `--feature-loss-layers "8,15,22,29"`).
- **Per-layer weighting** (added 2026-05-11): currently we average per-layer
  losses uniformly. fastai's recipe used `[5, 15, 2]` for the three default
  layers (relu2_2 / relu3_3 / relu4_3) to emphasize mid-level texture/edge
  features over the deepest semantic layer. Enable via:
  `--feature-content-layer-weights "5,15,2" --feature-style-layer-weights "5,15,2"`.
  Note: weights are raw multipliers, summed not averaged. fastai's weights
  sum to 22 (vs uniform 1.0), so the effective content/style magnitude is
  ~22× higher; compensate by lowering `--feature-content-weight` and
  `--feature-style-weight` to ~1/22 of current values OR accept the larger
  magnitude (the safeguards will skip steps if it spikes).

Predictions:
- If exp15 val LPIPS-squeeze < exp10's 0.153 by > 3% → Gram texture
  matching was a real bottleneck. Texture-aware loss is the way forward
  for stylization tasks.
- If exp15 ≈ exp10 → LPIPS already captured what Gram does for our
  task (unlikely but possible). Stay with simpler LPIPS aux.
- Visual check is more informative than the LPIPS number here — the
  expected gain is "stronger anime stylization" which may not score on
  pairwise LPIPS but is the qualitative win.
