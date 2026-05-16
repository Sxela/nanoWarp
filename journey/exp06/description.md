### exp06 — same run + every safety net we had (20k steps) — KILLED

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 --val-batches 4 `
    --prediction-type v --source-dropout 0.15 `
    --high-t-warmup-steps 2000 --high-t-warmup-low 500 `
    --grad-clip-norm 1.0 --lr-warmup-steps 500 --lr-cosine --lr-min 1e-5 `
    --outdir out/exp06_vpred_dropout_warmup_20k
```

- Stack: v-prediction + source dropout 0.15 + high-t warmup 2k + grad clip 1.0
  + LR warmup 500 + cosine decay to 1e-5.
- **Collapsed at the same step ~5000–6000** as exp05. Optimizer hygiene did
  not save it. Killed.
- This was the data point that pinned the cause on the trainable ResNet
  encoder layers (layer2/3/4 drift), not the optimizer.
