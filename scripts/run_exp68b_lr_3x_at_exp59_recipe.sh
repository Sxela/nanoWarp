#!/usr/bin/env bash
# exp68b — exp59 recipe with lr=6e-4 (3x default). Mid-aggressive LR.
# Warmup bumped to 1500 (3x) for proportional early ramp.
#
# Upper edge of what flow-matching repos commonly use; above this we're
# in "needs careful tuning" territory. Single-flag-set delta vs exp59:
#   --lr 6e-4 --lr-warmup-steps 1500
#
# A/B target — exp59 (val_portraits): face_lpips_sq=0.122
# If 68a (2x) wins and 68b (3x) also wins → LR ceiling is higher than 3x.
# If 68a wins and 68b ties/loses → 2x is the sweet spot.
# If both lose → LR isn't the bottleneck.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp68b_lr_3x_at_exp59_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp68b] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp68b_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp68b] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp68b_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp68b_model_best.pt"
        echo "[exp68b] resuming from $OUTDIR/exp68b_model_best.pt"
    else
        echo "[exp68b] no prior checkpoint, fresh start"
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
    --lr 6e-4 --lr-min 1e-5 --lr-warmup-steps 1500 \
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
    --exp-name exp68b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp68b_lr_3x_at_exp59_recipe \
    --wandb-tags "exp68b,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,lr_6e-4,lr_3x,warmup_1500,ablation_vs_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp68b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp68b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp68b_final_256px_val" \
    2>&1 | tee "out/val_exp68b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp68b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp68b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp68b_final_256px_val_portraits.log"

echo "[exp68b] done. A/B target — exp59 (val_portraits): face_lpips_sq=0.122"
