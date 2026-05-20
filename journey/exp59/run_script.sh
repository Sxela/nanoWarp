#!/usr/bin/env bash
# exp59 — exp50 recipe with cross-attention conditioning ADDED at the
# H/8 decoder level (alongside existing FiLM).
#
# Motivation: FiLM is per-channel γ,β scaling — every spatial position
# in the decoder gets the same gain/shift from the matching pyramid
# location. Cross-attention lets every decoder position query EVERY
# pyramid position, so it can pull source info from non-local positions
# (e.g. a chin landmark informing a forehead prediction). More expressive
# than FiLM at the cost of quadratic-in-tokens compute.
#
# Placement: H/8 decoder level only (32x32 = 1024 tokens at 256px input).
# That's the deepest non-bottleneck decoder level — where pyramid feature
# f3 (c4=352 channels) aligns with decoder activation (c3=352 channels).
# Cheaper resolutions (H/4=64x64=4096 tokens, H/2=128x128, H=256x256) are
# too expensive for full cross-attn; FiLM stays at those.
#
# Added params: ~500k (vs 50M base) -> 1% increase. Zero-init output
# projection makes it identity-at-init: safe insertion alongside FiLM,
# older checkpoints (no cross-attn) auto-detect from state_dict in ckpt.py.
#
# Single-flag delta vs exp50: --use-cross-attn-cond.
#
# A/B target: exp50 (val_portraits): face_lpips_sq=0.124, face_ssim=0.544
# 20k for cheap A/B; if it wins, promote to 80k + try at additional
# decoder levels.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp59_cross_attn_cond_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp59] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp59_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp59] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp59_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp59_model_best.pt"
        echo "[exp59] resuming from $OUTDIR/exp59_model_best.pt"
    else
        echo "[exp59] no prior checkpoint, fresh start"
    fi
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    $RESUME_ARG \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --use-cross-attn-cond \
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
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp59 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp59_cross_attn_cond_at_exp50_recipe \
    --wandb-tags "exp59,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,decoder_attn,source_pyramid,film,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp59] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp59_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp59_final_256px_val" \
    2>&1 | tee "out/val_exp59_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp59_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp59_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp59_final_256px_val_portraits.log"

echo "[exp59] done. A/B target — exp50 (val_portraits): face_lpips_sq=0.124, face_ssim=0.544"
