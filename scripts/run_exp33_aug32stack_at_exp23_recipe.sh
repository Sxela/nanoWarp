#!/usr/bin/env bash
# exp33 — exp23 recipe (20k @ 256px, bs=4) with the full exp32 aug stack.
# Clean A/B vs exp23 (lpips_sq=0.127) to isolate aug impact at fixed compute.
#
# Deltas vs exp23:
#   - aug stack: scale U[1.0, 2.5], rotate ±25°, perspective 0.15,
#     hflip, source color jitter ±0.3, source degradation
#     (downsample-up p=0.3, blur σ U[0.5,3.0] p=0.7, JPEG U[30,95] p=0.7),
#     clean_prob=0.2.
#   - script: train_exp32_prog512.py with phase1 skipped, capped at 256px.
# Everything else (mc=88, attn 16/32/64, no source encoder, flow FM,
# LPIPS-VGG weight 0.2, bf16, lr 2e-4→1e-5 cosine, warmup 500) matches exp23.

set -euo pipefail
# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

# Prepend repo root to PYTHONPATH (don't skip if PYTHONPATH is already set —
# Colab pre-populates it, which would shadow `src/` otherwise).
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
# WANDB_API_KEY must be set in the launching shell (e.g. via ~/.netrc or
# `export WANDB_API_KEY=...` before invoking this script). Never commit keys.
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp33_aug32stack_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 2.5 \
    --aug-rotate-deg 25.0 \
    --aug-perspective 0.15 --aug-perspective-prob 0.5 \
    --aug-brightness 0.3 --aug-contrast 0.3 --aug-saturation 0.3 \
    --clean-prob 0.2 \
    --degrade-resize-prob 0.3 --degrade-resize-min 0.25 --degrade-resize-max 0.75 \
    --corrupt-blur-max 3.0 --corrupt-blur-prob 0.7 \
    --corrupt-jpeg-min 30 --corrupt-jpeg-prob 0.7 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --sample-steps 20 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp33_aug32stack_at_exp23_recipe \
    --wandb-tags "exp33,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,exp32_aug_stack,ablation_vs_exp23" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp33] training done. Running final val (25 batches, EMA, sample_steps=20)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp33_final_256px" \
    2>&1 | tee "out/val_exp33_final_256px.log"

echo "[exp33] done. Compare lpips_vgg vs exp23 (out/val_exp23_final_256px/val_metrics.json)."
