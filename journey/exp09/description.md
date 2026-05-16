### exp09 — exp08-lpips + pixel_shuffle (encoder on, mc=56) — DONE

Tests sub-pixel conv upsampling on the **proven** architecture (exp08-lpips:
encoder on, FM, LPIPS 0.2). Single-variable change vs exp08-lpips:
`--upsample-type pixel_shuffle` instead of resize+conv. ICNR init applied to
avoid checkerboard artifacts at training start.

Param-matched to exp08-lpips by lowering `--model-ch 64 → 56`, since
PixelShuffle's `channels → 4·channels` upsamplers add ~21M params. At mc=56
with pixel_shuffle the total is **44.1M / 32.9M trainable**, within ~3% of
exp08-lpips's 42.7M / 31.6M.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --freeze-source-encoder all --model-ch 56 `
    --upsample-type pixel_shuffle `
    --source-dropout 0.15 --lpips-weight 0.2 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --num-workers 8 `
    --wandb --wandb-project nanoWarp `
    --wandb-run-name exp09_pixshuf_lpips_mc56_20k `
    --wandb-tags "flow,encoder-on,lpips,pixel-shuffle,exp09" `
    --outdir out/exp09_pixshuf_lpips_mc56_20k
```

Final val curve vs exp08-lpips (resize_conv, mc=64) at every checkpoint:

| step | exp08-lpips SSIM | **exp09 SSIM** | exp08-lpips LPIPS | **exp09 LPIPS** |
|---:|---:|---:|---:|---:|
|  1k | 0.608 | **0.583** | 0.161 | **0.170** |
|  5k | 0.678 | **0.655** | 0.148 | **0.153** |
| 10k | 0.703 | **0.685** | 0.148 | **0.153** |
| 15k | 0.715 | **0.700** | 0.150 | **0.154** |
| 20k | 0.719 | **0.707** | 0.152 | **0.155** |

**Result: pixel_shuffle strictly worse on both metrics at every step.** Gap
is small (~1.5% on SSIM, ~2% on LPIPS) but persistent across the whole run.
Caveat: the param match wasn't perfect — exp09 had 32.9M trainable
(vs exp08-lpips's 31.6M), so exp09 was slightly *larger* and still lost.
The "smaller model lost" explanation is ruled out.

What this tells us:
- The fastai-era PixelShuffle intuition doesn't carry. Probably because
  LPIPS aux 0.2 is already pushing for perceptual sharpness; the marginal
  contribution of "learnable single-step upsample" disappears.
- The 4× param cost in PixelShuffle's upsamplers got reallocated to width
  (64 → 56). Wider channels at every level beat sharper upsamplers at lower
  width on this task.
- ICNR init worked: losses are smooth, no checkerboard catastrophe. The
  implementation is fine — the technique just doesn't pay off here.

**Decision: resize_conv stays the default.** Flag remains available
(`--upsample-type pixel_shuffle`) for future tasks where the trade-off may
flip (e.g. higher resolution, no aux loss).
