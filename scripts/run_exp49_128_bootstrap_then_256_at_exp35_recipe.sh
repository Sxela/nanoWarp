#!/usr/bin/env bash
# exp49 — 1k @ 128px bootstrap, then 19k @ 256px convergence. exp35 arch,
# minimal aug, constant LPIPS. Direct A/B vs exp35 (0k @ 128 + 20k @ 256)
# to answer "does the 128 bootstrap help final 256 quality?".
#
# Identical to exp35 except 1k of the 20k budget is spent at 128px first.
# All other knobs match exp35 for clean comparison.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp49_128_bootstrap_then_256_at_exp35_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 1000 \
    --phase2-end 20000 \
    --bs-128 16 --bs-256 4 --bs-512 4 \
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
    --exp-name exp49 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp49_128_bootstrap_then_256_at_exp35_recipe \
    --wandb-tags "exp49,ds1k,prog128_256,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,128_bootstrap,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp49] training done. Running final val @ 256..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp49_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp49_final_256px" \
    2>&1 | tee "out/val_exp49_final_256px.log"

echo "[exp49] done. Compare vs exp35 (no 128 bootstrap):"
echo "  exp35:  face_lpips_sq=0.1526  face_lpips_vgg=0.2859  face_ssim=0.728  lpips_vgg=0.2395"
echo "  exp49:  <fresh> — does 1k @ 128 bootstrap help final 256 quality?"
