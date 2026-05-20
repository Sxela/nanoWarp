#!/usr/bin/env bash
# exp68a — exp59 recipe (cross-attn @ H/8, minimal aug, exp35 arch,
# 20k @ 256px bs=4) with lr=4e-4 (2x the long-standing 2e-4 default).
# Warmup bumped to 1000 (2x) to keep the early ramp gentle.
#
# Motivation: 2e-4 has been the default since exp01 era and was never
# re-tuned for the modern arch (cross-attn, pyramid, FiLM). The recent
# "no signal at 20k" pattern (exp64/66/67 all TIE) could be partially
# "optimizer not moving fast enough" — high-LR test addresses that
# directly.
#
# Safe variant: 2x LR is lit-typical for "I want to train faster"
# bumps. grad_clip=1.0 unchanged.
#
# Single-flag-set delta vs exp59: --lr 4e-4 --lr-warmup-steps 1000.
#
# A/B target — exp59 (val_portraits): face_lpips_sq=0.122
# Wins (≥3% face_lpips_sq improvement) at 20k → LR was the bottleneck;
# rerun exp52/60 recipes at 80k with the higher LR.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp68a_lr_2x_at_exp59_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp68a] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp68a_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp68a] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp68a_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp68a_model_best.pt"
        echo "[exp68a] resuming from $OUTDIR/exp68a_model_best.pt"
    else
        echo "[exp68a] no prior checkpoint, fresh start"
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
    --lr 4e-4 --lr-min 1e-5 --lr-warmup-steps 1000 \
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
    --exp-name exp68a \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp68a_lr_2x_at_exp59_recipe \
    --wandb-tags "exp68a,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,lr_4e-4,lr_2x,warmup_1000,ablation_vs_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp68a] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp68a_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp68a_final_256px_val" \
    2>&1 | tee "out/val_exp68a_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp68a_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp68a_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp68a_final_256px_val_portraits.log"

echo "[exp68a] done. A/B target — exp59 (val_portraits): face_lpips_sq=0.122"
