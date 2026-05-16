#!/usr/bin/env bash
# exp40 — exp35 recipe (pyramid + decoder attn + minimal aug) + VGG style loss.
#
# Adds a Gatys/Johnson-style Gram-matrix loss on top of LPIPS, computed from
# VGG16 features at relu2_2 / relu3_3 / relu4_3 (layers 8/15/22). Pushes the
# output toward target *texture statistics* independent of pixel alignment —
# complements LPIPS which is pixel-aligned. Should reward "looks like the
# comic style" beyond what LPIPS captures.
#
# Final validate.py runs with the standard (no-CFG) sampler.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp40_vgg_style_at_exp35_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --style-loss-weight 5000.0 --content-loss-weight 0.0 \
    --style-loss-layers "8,15,22" \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --sample-steps 20 \
    --exp-name exp40 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp40_vgg_style_at_exp35_recipe \
    --wandb-tags "exp40,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,vgg_style_loss,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp40] training done. Running final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp40_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp40_final_256px" \
    2>&1 | tee "out/val_exp40_final_256px.log"

echo "[exp40] done. Compare vs exp35:"
echo "  exp35:  face_lpips_sq=0.1526  face_lpips_vgg=0.2859  lpips_vgg=0.2395"
echo "  exp40:  <fresh>"
