#!/usr/bin/env bash
# exp67 — exp59 recipe (cross-attn @ H/8, minimal aug, exp35 arch,
# 20k @ 256px bs=4) with SGDR 2-cycle warm-restart LR schedule.
#
# Default cosine: smooth lr_max -> lr_min over the full post-warmup
# window. The model spends the last ~30% of training near lr_min
# (~1e-5) which often means it's stuck at the local minimum it found
# in the first half and can't escape.
#
# SGDR 2-cycle: split the post-warmup window into 2 equal cycles, each
# decaying lr_max -> lr_min. The lr "warm restart" at the midpoint
# kicks the optimizer out of the local minimum and lets it find a
# (hopefully better) one.
#
# Lit precedent: Loshchilov & Hutter 2016 (SGDR), used widely in modern
# diffusion/flow recipes. Reported 1-3% improvement when training
# plateaus, which is consistent with what our exp52/exp60 val curves
# show late in training.
#
# Single-flag delta vs exp59: --lr-num-cycles 2.
#
# A/B target — exp59 (val_portraits): face_lpips_sq=0.122
#
# If wins clearly at 20k, the natural follow-up is to apply 2-cycle to
# exp61's 80k recipe as the new deployment canonical. exp52's
# late-training plateau is more pronounced at 80k than at 20k, so the
# upside scales.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp67_sgdr_2cycle_at_exp59_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp67] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp67_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp67] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp67_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp67_model_best.pt"
        echo "[exp67] resuming from $OUTDIR/exp67_model_best.pt"
    else
        echo "[exp67] no prior checkpoint, fresh start"
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
    --lr-num-cycles 2 --lr-cycle-mult 1.0 \
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
    --exp-name exp67 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp67_sgdr_2cycle_at_exp59_recipe \
    --wandb-tags "exp67,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,sgdr_2cycle,decoder_attn,source_pyramid,film,ablation_vs_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp67] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp67_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp67_final_256px_val" \
    2>&1 | tee "out/val_exp67_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp67_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp67_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp67_final_256px_val_portraits.log"

echo "[exp67] done. A/B target — exp59 (val_portraits): face_lpips_sq=0.122"
echo "  Look at wandb LR plot — should show 2 cosine cycles, restarting at step 10250."
