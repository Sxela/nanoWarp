#!/usr/bin/env bash
# exp34 — exp33 recipe (20k @ 256px, bs=4, full exp32 aug stack) + symmetric
# decoder spatial self-attention (mirrors encoder attn).
#
# Clean A/B vs exp33 (same data, same aug, same compute) to isolate the
# contribution of putting BottleneckAttention on the decoder side at the
# same resolutions as the encoder (32, 64 for attn_resolutions=16,32,64;
# 16 is mid_attn, always on; 256/128 levels are not in the set).
#
# Architecture delta vs exp33:
#   - attn_dec4 at H/8 (32px), channels c3=352 — mirrors attn4
#   - attn_dec3 at H/4 (64px), channels c3=352 — mirrors attn3
#   - attn_dec2 at H/2 (128px) — None (128 not in attn_set)
#   - attn_dec1 at H   (256px) — None (256 not in attn_set)
# Applied after each dec* ResBlock, before FiLM/tattn (when those exist).
#
# Original asymmetry note: nanoWarp's UNet had spatial self-attn only in the
# encoder + bottleneck; the decoder never had it. SD/SDXL puts it
# symmetrically. exp34 tests whether closing that asymmetry helps.
#
# No inference-time external dependencies.

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

OUTDIR=out/exp34_decoder_attn_noenc_attn163264_bf16_mc88_256px_20k
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
    --wandb-run-name exp34_decoder_attn_at_exp33_recipe \
    --wandb-tags "exp34,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,exp32_aug_stack,decoder_attn,ablation_vs_exp33" \
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
