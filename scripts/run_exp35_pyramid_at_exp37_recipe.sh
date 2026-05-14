#!/usr/bin/env bash
# exp35 — exp37 recipe (minimal aug + decoder attn) + source pyramid + FiLM.
#
# Stacks the pyramid on top of exp37's baseline (decoder attn, exp23-equivalent
# minimal aug). Targets fine-feature reconstruction (faces are the main pain
# point at this dataset size); the pyramid gives the decoder per-pixel source
# guidance at every resolution, including full-res, which the source-in-stem
# path doesn't deliver beyond the in_conv layer.
#
# Architecture delta vs exp37:
#   - SourcePyramid: 4-stage conv pyramid on raw source, features at the
#     4 decoder resolutions, ~1.8M params.
#   - FiLM: per-level 1x1 conv produces (γ, β); decoder activation becomes
#     x * (1 + γ) + β. Both γ and β zero-init → identity at init.
#   - Total added params: ~2.4M.
#
# No inference-time external dependencies.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp35_pyramid_at_exp37_recipe_noenc_attn163264_bf16_mc88_256px_20k
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
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --sample-steps 20 \
    --exp-name exp35 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp35_pyramid_at_exp37_recipe \
    --wandb-tags "exp35,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,ablation_vs_exp37" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp35] training done. Running final val (25 batches, EMA, sample_steps=20)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp35_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp35_final_256px" \
    2>&1 | tee "out/val_exp35_final_256px.log"

echo "[exp35] done. Compare vs:"
echo "  exp23  (clean, no arch changes):       lpips_vgg=0.234  ssim=0.689"
echo "  exp37  (decoder attn, clean):          lpips_vgg=0.242  ssim=0.684  Δlpips=0.133"
echo "  exp35  (decoder attn + pyramid):       <fresh> — looking for face quality + style sharpness"
