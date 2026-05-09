# First experiments to run

Assume your dataset root is available as:

```bash
DATASET=/path/to/photo2comics_dataset
```

And your layout is:

```text
$DATASET/
  train/
    source/
    target/
  val/
    source/
    target/
```

## 1. Baseline

```bash
python3 scripts/train.py img2img-v1 "$DATASET" --config configs/photo2comics_baseline.yaml --outdir out/exp01_baseline
```

## 2. Validate baseline with EMA

```bash
python3 scripts/sample.py img2img-v1-val "$DATASET" --config configs/photo2comics_baseline.yaml --checkpoint out/exp01_baseline/model.pt --use-ema --outdir out/exp01_baseline_val
```

## 3. Fast inference from baseline (20 steps)

```bash
python3 scripts/sample.py img2img-v1-infer "$DATASET" --config configs/photo2comics_baseline.yaml --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 20 --outdir out/exp01_baseline_infer20
```

## 4. Source also enters input stem

```bash
python3 scripts/train.py img2img-v1 "$DATASET" --config configs/photo2comics_baseline.yaml --source-in-stem --outdir out/exp02_source_in_stem
```

## 5. Add LPIPS auxiliary train loss

```bash
python3 scripts/train.py img2img-v1 "$DATASET" --config configs/photo2comics_baseline.yaml --lpips-weight 0.1 --outdir out/exp03_lpips_01
```

## 6. Source-in-stem + LPIPS

```bash
python3 scripts/train.py img2img-v1 "$DATASET" --config configs/photo2comics_baseline.yaml --source-in-stem --lpips-weight 0.1 --outdir out/exp04_stem_plus_lpips
```

## 7. Faster inference sweep

```bash
python3 scripts/sample.py img2img-v1-infer "$DATASET" --config configs/photo2comics_baseline.yaml --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 5  --outdir out/exp01_infer05
python3 scripts/sample.py img2img-v1-infer "$DATASET" --config configs/photo2comics_baseline.yaml --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 20 --outdir out/exp01_infer20
python3 scripts/sample.py img2img-v1-infer "$DATASET" --config configs/photo2comics_baseline.yaml --checkpoint out/exp01_baseline/model.pt --use-ema --sample-steps 50 --outdir out/exp01_infer50
```

## What to compare first

- validation loss
- validation SSIM
- validation LPIPS
- training panels
- validation panels
- fast inference panels at 5 / 20 / 50 steps

