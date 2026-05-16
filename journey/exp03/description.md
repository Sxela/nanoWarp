### exp03 — LPIPS aux loss 0.1 (2k steps)

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml --lpips-weight 0.1 `
    --outdir out/exp03_lpips_01
```

- LPIPS aux dropped from 0.87 → ~0.35 over the run. Did not validate further;
  pivoted to longer baselines after this.
