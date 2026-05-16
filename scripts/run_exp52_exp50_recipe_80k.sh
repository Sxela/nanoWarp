#!/usr/bin/env bash
# exp52 — exp50 (3k mixed) extended to 80k steps.
#
# exp50 at 20k beat every legacy baseline on val_portraits (face_lpips_sq
# 0.124 vs exp25-80k's 0.169) and stays competitive on legacy val (face
# regression mild vs exp51's catastrophic). Natural next move: train it
# longer and find the new 3k-data ceiling.
#
# Mirrors the legacy exp23 → exp25 pattern (20k → 80k on same recipe
# gave lpips_sq 0.128 → 0.115 on legacy val). Same architecture as
# exp50 (decoder attn + pyramid), same data, just 4× the training.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp52_exp50_recipe_80k_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    --steps 80000 \
    --phase1-end 0 \
    --phase2-end 80000 \
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
    --val-every 5000 --panel-every 5000 --checkpoint-every 10000 --best-every 5000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp52 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp52_exp50_recipe_80k \
    --wandb-tags "exp52,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,80k,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp52] training done. Validating on legacy val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp52_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp52_final_256px_val" \
    2>&1 | tee "out/val_exp52_final_256px_val.log"

echo "[exp52] validating on val_portraits..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp52_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp52_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp52_final_256px_val_portraits.log"

echo "[exp52] done. Compare on val_portraits:"
echo "  exp50 (20k, 3k mixed):  face_lpips_sq=0.124  face_lpips_vgg=0.285  face_ssim=0.544"
echo "  exp51 (20k, FFHQ-only): face_lpips_sq=0.122  face_lpips_vgg=0.280  face_ssim=0.550"
echo "  exp52 (80k, 3k mixed):  <fresh — does 4× training on mixed close the gap to FFHQ-only?>"
echo "  ...and on legacy val (small/peripheral faces): exp50=0.201, exp51=0.290. Does exp52 hold or regress?"
