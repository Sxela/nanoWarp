### exp02 — source-in-stem (2k steps)

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml --source-in-stem `
    --outdir out/exp02_source_in_stem
```

- Loss 0.026 (better than exp01 0.031), but SSIM 0.230 (worse) and LPIPS 0.689 (worse).
- Source-in-stem helps the random-t reconstruction loss but hurts structural
  similarity at this scale.
