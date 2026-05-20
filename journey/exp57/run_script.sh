#!/usr/bin/env bash
# exp57 — exp50 recipe (3k mixed, flow, exp35 arch, 20k @ 256px bs=4,
# LPIPS=0.2 vgg, minimal aug) with --source-dropout 0.2 as pure
# training-time regularization.
#
# Critical distinction from exp41: this is regularization-only, NOT CFG.
#   - --source-dropout 0.2: 20% of training batch elements get their
#     source channels zeroed. Model must predict target from noise +
#     time only for those samples. Forces it to learn priors about the
#     target distribution.
#   - --cfg-scale 1.0 at inference (default): single forward pass per
#     step, conditioned only. No CFG amplification. Flow + CFG cratered
#     in exp41 (ssim 0.36 at scale=2.0) because v is a true velocity —
#     so we don't go near it.
#
# Why it might help at 3k pairs + 80k steps (exp52 territory):
# - 3k pairs * 80k steps / bs=4 = 100+ epochs → real over-memorization
#   risk on source→target shortcuts.
# - Implicit target-distribution prior makes the model more robust when
#   the source signal is weak (corrupted, low-light, OOD).
#
# A/B target: exp50 (20k, no source dropout) on val_portraits:
#   face_lpips_sq=0.124, face_lpips_vgg=0.285, face_ssim=0.544
# 20k for cheap A/B; if it wins, promote to 80k as exp57b vs exp52.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp57_source_dropout02_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp57] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp57_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp57] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp57_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp57_model_best.pt"
        echo "[exp57] resuming from $OUTDIR/exp57_model_best.pt"
    else
        echo "[exp57] no prior checkpoint, fresh start"
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
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 \
    --source-dropout 0.2 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp57 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp57_source_dropout02_at_exp50_recipe \
    --wandb-tags "exp57,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,source_dropout_02,no_cfg,decoder_attn,source_pyramid,film,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp57] training done. Final val (no CFG; default --cfg-scale 1.0)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp57_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp57_final_256px_val" \
    2>&1 | tee "out/val_exp57_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp57_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp57_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp57_final_256px_val_portraits.log"

echo "[exp57] done. A/B target — exp50 (val_portraits): face_lpips_sq=0.124, face_ssim=0.544"
