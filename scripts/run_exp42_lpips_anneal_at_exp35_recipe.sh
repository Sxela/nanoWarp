#!/usr/bin/env bash
# exp42 — exp35 recipe (pyramid + decoder attn + minimal aug) but with LPIPS
# weight cosine-annealed from 0.2 → 0.0 over 20k steps.
#
# Hypothesis: LPIPS-VGG's L2-in-feature-space rewards mode-averaging — outputs
# converge toward a "feature centroid" of plausible targets rather than
# committing to specific stylistic patterns (anime-eye shapes, sharp contour
# lines). Without LPIPS the model never even reaches the style manifold
# (confirmed in earlier experiments). The schedule: full LPIPS early to
# get the model onto the right manifold, then anneal to pure flow-matching
# MSE so fine specifics can sharpen without the averaging pressure.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp42_lpips_anneal_at_exp35_recipe_noenc_attn163264_bf16_mc88_256px_20k
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
    --lpips-weight 0.2 --lpips-weight-end 0.0 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --sample-steps 20 \
    --exp-name exp42 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp42_lpips_anneal_at_exp35_recipe \
    --wandb-tags "exp42,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,lpips_anneal,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp42] training done. Running final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp42_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp42_final_256px" \
    2>&1 | tee "out/val_exp42_final_256px.log"

echo "[exp42] done. Compare vs exp35:"
echo "  exp35 (const lpips=0.2):     face_lpips_sq=0.1526  face_lpips_vgg=0.2859  lpips_vgg=0.2395"
echo "  exp42 (anneal 0.2 → 0.0):    <fresh> — look for sharper face features in panels"
