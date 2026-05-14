#!/usr/bin/env bash
# exp36 — exp33 recipe (20k @ 256px, bs=4, full exp32 aug stack) + DiT
# bottleneck.
#
# Architecture delta vs exp33: the conv bottleneck (mid_attn + mid2 ResBlock)
# is replaced by 4 DiT-XL-style transformer blocks operating on the flattened
# (H/16 × W/16, cm=704) token grid. `mid1` still projects c4 → cm upstream so
# the DiT stack always operates at constant width.
#
# Each DiT block: adaLN-zero conditioning from t_emb → MHSA → adaLN-zero
# conditioning → MLP. adaLN gates zero-init → block emits its input unchanged
# at step 0, so a no-DiT checkpoint loads cleanly via strict=False.
#
# Param budget: +~28M over exp33 (49M → ~77M total). This is well outside
# "same param budget" territory — the experiment is testing whether the
# capacity is worth the cost.
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

OUTDIR=out/exp36_dit_bottleneck_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-dit-bottleneck \
    --num-dit-blocks 4 \
    --dit-mlp-ratio 4.0 \
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
    --exp-name exp36 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp36_dit_bottleneck_at_exp33_recipe \
    --wandb-tags "exp36,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,exp32_aug_stack,dit_bottleneck,ablation_vs_exp33" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp36] training done. Running final val (25 batches, EMA, sample_steps=20)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp36_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp36_final_256px" \
    2>&1 | tee "out/val_exp36_final_256px.log"

echo "[exp36] done. Compare lpips_vgg vs exp33 (out/val_exp33_final_256px/val_metrics.json)."
