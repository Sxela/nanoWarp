## exp32 — train from scratch, progressive 128→256→512px, full augmentation

**Motivation**: exp31 showed fine-tuning from exp25 on corrupted inputs degrades clean-val
(0.1447→0.1824). Training from scratch with progressive resolution and rich augmentation
should produce a model that is simultaneously robust to real-video compression *and*
high quality on clean sources, since it never overfit to clean-only training first.

**Architecture**: identical to exp25 — mc=88, no source encoder (source in stem),
attn_res=(16,32,64), flow FM, LPIPS-VGG weight 0.2, bf16 AMP.

**Progressive phases**:

| Phase | Steps | Resolution | BS | Effective 512px-equivalent steps |
|-------|-------|------------|-----|----------------------------------|
| 1     | 5k    | 128px      | 64  | ~80k (16× area ratio)            |
| 2     | 20k   | 256px      | 16  | ~80k (4× area ratio)             |
| 3     | 75k   | 512px      | 4   | 75k                              |

**Augmentation** (per sample, randomized):

Shared geometry (source + target):
- Zoom scale ~ U[1.0, 2.5] → resize → random crop
- Rotation ±25°
- Perspective warp distortion=0.15, p=0.5
- Horizontal flip p=0.5

Source-only color jitter: brightness/contrast/saturation ±0.3

Source-only degradation (80% of samples, gated by clean_prob=0.2):
- Resize-down+up p=0.3 (factor ~ U[0.25, 0.75]) — "internet/compression" pixelation
- Gaussian blur σ~U[0.5,3.0] p=0.7
- JPEG quality~U[30,95] p=0.7

**LR**: 2e-4 → 1e-6 cosine over 100k steps, warmup 500 steps.
**Val/checkpoints**: val+panel+nat1 every 5k, checkpoint every 10k, best saved on val LPIPS.

```bash
OUTDIR=out/exp32_prog512_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTDIR
PYTHONPATH=. \
TORCH_HOME=/tmp/torch_home \
WANDB_API_KEY=$WANDB_API_KEY \
WANDB_CACHE_DIR=/tmp/wandb_cache \
WANDB_CONFIG_DIR=/tmp/wandb_config \
MPLCONFIGDIR=/tmp/mplconfig \
python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --wandb --wandb-run-name exp32_prog512 \
    --outdir $OUTDIR \
    2>&1 | tee $OUTDIR/train.log
```

Outdir: `out/exp32_prog512_20260514_033045/`
Wandb: https://wandb.ai/alx-spirin/nanoWarp/runs/wes3p2ce

Results: TBD (run in progress, 100k steps)

---
