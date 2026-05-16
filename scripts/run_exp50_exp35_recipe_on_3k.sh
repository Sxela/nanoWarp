#!/usr/bin/env bash
# exp50 — exp35 recipe (pyramid + decoder attn + minimal aug, constant
# LPIPS=0.2) run on the new 3k merged dataset (original 1k + 2.3k FFHQ).
#
# Key change vs exp35: training data is 3.2× larger AND now includes real
# photo sources (the original 1k had Flux-generated synthetic sources). Two
# stacking effects:
#   1. More data → architectural ceiling lifts.
#   2. Real-source domain finally in training → model stops being OOD on
#      real photos at inference (nat1.mp4, phone photos, etc.).
#
# Validation split structure now has two:
#   val/           — original 100 group photos (continuity with all prior runs)
#   val_portraits/ — 200 FFHQ portraits (the meaningful face-quality signal)
# Both validated at the end; both logged to the same wandb run with
# separate key prefixes.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp50_exp35_recipe_on_3k_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
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
    --panel-keys "000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp50 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp50_exp35_recipe_on_3k \
    --wandb-tags "exp50,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,data_scale_3k,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp50] training done. Running final val on legacy 'val' split..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp50_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp50_final_256px_val" \
    2>&1 | tee "out/val_exp50_final_256px_val.log"

echo "[exp50] running final val on val_portraits (FFHQ-style face crops)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp50_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp50_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp50_final_256px_val_portraits.log"

echo "[exp50] done. Compare vs exp35 (1k dataset, same recipe):"
echo "  exp35 (1k, val):                face_lpips_sq=0.1526  face_ssim=0.728  lpips_vgg=0.2395"
echo "  exp50 (3k, val):                <fresh — apples-to-apples on legacy val>"
echo "  exp50 (3k, val_portraits):      <fresh — meaningful face-quality signal>"
