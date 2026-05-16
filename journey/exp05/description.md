### exp05 — long baseline (20k steps, no safety nets) — KILLED

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --steps 20000 --log-every 100 --panel-every 1000 --val-every 1000 `
    --outdir out/exp05_long_baseline_20k
```

- Diverged sharply at ~step 5000–6000. Loss floor jumped from ~0.02 to ~0.15
  and stayed elevated. Visible in panels: x0_hat clean through step 4k, grainy
  from step 5k+. Killed at step ~7000.
