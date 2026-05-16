### exp07b — same as exp07 but with mid-run checkpoints + full-sample panels (20k steps) — DONE

After adding `--checkpoint-every` and replacing the random-t panel with a
full-sample panel:

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --method flow --flow-sigma-noise 0.05 `
    --freeze-source-encoder all --source-dropout 0.15 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --checkpoint-every 1000 --sample-panel-steps 20 `
    --outdir out/exp07b_flow_chkpt_20k
```

Mid-run validations (run in parallel as checkpoints land):

```powershell
python scripts/sample.py img2img-v1-val data/photo2anime `
    --checkpoint out/exp07b_flow_chkpt_20k/model_step_001000.pt --use-ema `
    --sample-steps 20 --max-batches 4 --panel-count 2 `
    --outdir out/exp07b_val_step_001000

# step-5000 watcher armed; will fire automatically when that checkpoint lands.
```

Val curve across the run:

| step | loss | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|
|  1k | 0.0177 | 0.616 | 0.219 |
|  5k | 0.0100 | 0.629 | 0.176 |
| 10k | 0.0029 | 0.648 | 0.190 |
| 15k | 0.0026 | 0.691 | 0.166 |
| 20k | 0.0025 | 0.701 | 0.159 |

- MSE loss had nearly plateaued by 15k, but **SSIM and LPIPS were still
  improving** at 20k. Pixel-MSE convergence is independent of perceptual
  quality at this scale — the model was still refining detail that doesn't
  show up in MSE.
- Visually, the predicted-target column was still mostly
  "source + mild palette shift" through step 4k; meaningful anime stylization
  was emerging slowly. Real flat-color stylization didn't really kick in
  until LPIPS aux was added (see exp07b+LPIPS below).
