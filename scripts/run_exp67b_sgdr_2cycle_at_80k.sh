#!/usr/bin/env bash
# exp67b — SGDR 2-cycle LR at 80k.
#
# exp67 @ 20k vs exp59 was a clean TIE across the board (face_lpips_sq
# 0.122 exact tie). SGDR's mechanism is plateau escape via warm restart
# — at 20k the cosine baseline doesn't plateau enough for the restart
# to matter.
#
# At 80k the late-training plateau is real and visible in exp52/exp60
# wandb val curves (the curves visibly flatten over the last 30k steps).
# That's where SGDR is supposed to help.
#
# Single-flag delta vs exp60 recipe: --lr-num-cycles 2.
# (No code change — SGDR scheduler already supported in cosine_lr.)
#
# Cycle math at 80k with warmup=500:
#   Phase 1: warmup 0-500 → cosine 500 -> ~40000 (lr_max=2e-4 to lr_min=1e-5)
#   Phase 2: warm restart at ~40000 → cosine ~40000 -> 80000 (same schedule)
#
# A/B target — exp60 (val_portraits): face_lpips_sq=0.0997, ssim=0.460
#
# Decision rule:
# - exp67b face_lpips_sq < 0.095 → SGDR breaks the plateau; promote to
#   canonical, retest on the exp61 deployment recipe (exp67c).
# - 0.095 ≤ exp67b ≤ 0.103 → within noise of exp60; cheap-to-add lever
#   that doesn't hurt, but doesn't justify recipe change.
# - > 0.103 → SGDR actively destabilizes at 80k (warm restart hurts
#   late-training); document and park.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp67b_sgdr_2cycle_at_80k_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp67b] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp67b_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp67b] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp67b_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp67b_model_best.pt"
        echo "[exp67b] resuming from $OUTDIR/exp67b_model_best.pt"
    else
        echo "[exp67b] no prior checkpoint, fresh start"
    fi
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    $RESUME_ARG \
    --steps 80000 \
    --phase1-end 0 \
    --phase2-end 80000 \
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
    --val-every 5000 --panel-every 5000 --checkpoint-every 10000 --best-every 5000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp67b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp67b_sgdr_2cycle_at_80k \
    --wandb-tags "exp67b,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,sgdr_2cycle,decoder_attn,source_pyramid,film,80k,ablation_vs_exp60,plateau_escape_test" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp67b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp67b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp67b_final_256px_val" \
    2>&1 | tee "out/val_exp67b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp67b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp67b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp67b_final_256px_val_portraits.log"

echo "[exp67b] done. A/B target — exp60 (val_portraits): face_lpips_sq=0.0997"
echo "  Watch the wandb LR curve — should show two cosine cycles with restart at step ~40250."
