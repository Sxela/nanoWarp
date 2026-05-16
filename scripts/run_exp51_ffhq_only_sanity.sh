#!/usr/bin/env bash
# exp51 — sanity test: train exp35 recipe on FFHQ-only pairs.
#
# Question: if the model can't learn frontal-face stylization on this
# clean, uniform dataset (every pair is a centered FFHQ portrait paired
# with the same Flux-anime style), no amount of data mixing will help.
#
# This isolates "is the architecture+recipe capable of learning faces"
# from "does data mix help". Compare exp51's val_portraits numbers to
# exp35-on-3k (exp50) and the retroactive exp25-80k baseline.
#
# Dataset: data/photo2anime_ffhq2k/  (2321 train + 200 val == val_portraits)
#   - All FFHQ → Flux-anime, no original synthetic-source pairs.
#   - val/ == val_portraits/ (no separate legacy val).

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp51_ffhq_only_sanity_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_ffhq2k \
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
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370" \
    --sample-steps 20 \
    --exp-name exp51 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp51_ffhq_only_sanity \
    --wandb-tags "exp51,ds_ffhq2k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,sanity,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp51] training done. Validating on val_portraits..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_ffhq2k \
    --checkpoint "$OUTDIR/exp51_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp51_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp51_final_256px_val_portraits.log"

echo "[exp51] done. Compare on val_portraits:"
echo "  exp25 (80k, 1k synth):           face_lpips_sq=0.169  face_ssim=0.500"
echo "  exp35 (20k, 1k synth):           face_lpips_sq=0.178  face_ssim=0.477"
echo "  exp51 (20k, 2.3k FFHQ-only):     <fresh — does pure FFHQ training beat mixed?>"
