#!/usr/bin/env bash
# exp34 — exp33 recipe (20k @ 256px, bs=4, full exp32 aug stack) + in-model
# source feature pyramid + FiLM modulation of the decoder.
#
# Clean A/B vs exp33 (same data, same aug, same compute) to isolate the
# pyramid's contribution. Architecture delta:
#   - SourcePyramid: 4-stage conv pyramid on raw source, features at the
#     4 decoder resolutions, ~1.8M params.
#   - FiLM: per-level 1x1 conv produces (γ, β); decoder activation becomes
#     x * (1 + γ) + β. Both γ and β zero-init → identity at init.
#   - Total added params: ~2.4M at mc=88 (UNet stays ~49M; new total ~51M).
#
# No inference-time external dependencies. Pyramid runs once per source per
# forward pass (it's not t-conditioned, so it's redundant across ODE steps —
# acceptable cost for now, can be cached later if profiling demands it).

set -euo pipefail
cd "$(dirname "$0")/.."

# Prepend repo root to PYTHONPATH (Colab pre-populates it, which would shadow
# `src/` otherwise).
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp34_pyramid_film_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-source-pyramid \
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
    --wandb-run-name exp34_pyramid_film_at_exp33_recipe \
    --wandb-tags "exp34,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,exp32_aug_stack,source_pyramid,film,ablation_vs_exp33" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp34] training done. Running final val (25 batches, EMA, sample_steps=20)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp34_final_256px" \
    2>&1 | tee "out/val_exp34_final_256px.log"

echo "[exp34] done. Compare lpips_vgg vs exp33 (out/val_exp33_final_256px/val_metrics.json)."
