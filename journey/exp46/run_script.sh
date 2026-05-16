#!/usr/bin/env bash
# exp46 — compressed progressive 128 → 256 → 512 in a 20k total budget,
# on top of exp35 arch (decoder attn + pyramid) with LPIPS anneal 0.2 → 0.1
# (the exp45 floor that avoided the late-training MSE blur).
#
# Phase split (compute-balanced shape, just shorter than exp32's 100k):
#   1k @ 128px bs=16  → structure bootstrap
#   4k @ 256px bs=8   → detail learning
#   15k @ 512px bs=4  → high-res convergence
#
# Clean aug (exp35-style: minimal geometric, no corruption) so the
# architectural delta from "trained-at-512" is read without aug confound
# vs exp35 / exp45 which were 256-only.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp46_compressed_progressive_at_exp35_recipe_noenc_attn163264_bf16_mc88_prog128_256_512_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 1000 \
    --phase2-end 5000 \
    --bs-128 16 --bs-256 8 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-weight-end 0.1 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp46 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp46_compressed_progressive_at_exp35_recipe \
    --wandb-tags "exp46,ds1k,prog128_256_512,mc88,lpips_vgg,decoder_attn,source_pyramid,film,lpips_anneal_floor01,20k,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp46] training done. Running final val at 512px (training-final res)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp46_model.pt" \
    --image-size 512 --batch-size 2 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp46_final_512px" \
    2>&1 | tee "out/val_exp46_final_512px.log"

echo "[exp46] also validating at 256 for apples-to-apples vs exp35/exp45..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp46_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp46_final_256px" \
    2>&1 | tee "out/val_exp46_final_256px.log"

echo "[exp46] done. Compare vs:"
echo "  exp35  (20k @ 256, constant LPIPS):              face_lpips_sq=0.1526  face_ssim=0.728  lpips_vgg=0.2395"
echo "  exp45  (20k @ 256, LPIPS anneal floor 0.1):      face_lpips_sq=tied with exp35 in-loop"
echo "  exp44  (100k progressive, mid aug, anneal->0):   face_lpips_sq=0.154   face_ssim=0.726  lpips_vgg=0.238"
echo "  exp46  (20k compressed progressive @ 512 final): <fresh>"
