### exp01 — baseline (eps diffusion, partial freeze, 2k steps)

Reproduces docs/first_experiments.md step 1.

```powershell
python scripts/train.py img2img-v1 data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --outdir out/exp01_baseline
```

- 2000 steps, bs=4, 128px, lr=2e-4, EMA 0.999, ε prediction.
- Final loss ~0.057, val mean_loss ~0.012.
- Visually clean train-time x0_hat panels — but full DDIM-50 inference output was
  washed grey scribbles. Triggered the realization that train-time x0_hat at
  random t is not the same as full reverse sampling.

Validation (later replaced by the rewritten validate.py):

```powershell
python scripts/sample.py img2img-v1-val data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema `
    --outdir out/exp01_baseline_val
```

Sampler/clamp investigation runs on the same exp01 EMA checkpoint:

```powershell
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 5  `
    --outdir out/exp01_infer05
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 20 `
    --outdir out/exp01_infer20
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 50 `
    --outdir out/exp01_infer50
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 999 `
    --outdir out/exp01_infer999_ddim_clamp
python scripts/sample.py img2img-v1-infer data/photo2anime/val `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 1000 `
    --outdir out/exp01_infer1000_ddpm
```

All flavors (DDIM 5/20/50, DDIM 999 with clamp, full DDPM 1000) failed in
different ways. Concluded the issue was the **model**, not the sampler.

After rewriting validate.py to do full reverse sampling + high-t diagnostic:

```powershell
python scripts/sample.py img2img-v1-val data/photo2anime `
    --config configs/photo2comics_baseline.yaml `
    --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 50 `
    --max-batches 4 --panel-count 2 --save-progress-strip `
    --outdir out/exp01_baseline_val_v2
```

Result: `mean_loss=0.0095, mean_ssim_sampled=0.353, mean_lpips_sampled=0.533`.
The high-t diagnostic column showed pure colored static — confirmed source
conditioning collapses at high t.
