#!/usr/bin/env bash
# exp60 — 80k promotion of exp59's cross-attn win.
#
# exp59 (20k, +cross-attn @ H/8) gave a clean uniform win vs exp50:
#   face_lpips_sq portraits 0.124 -> 0.122 (-1.6%, best 20k flow ever)
#   uniform improvement across val_portraits, zero regressions on either split.
#
# Question: does the architectural win at 20k hold up at 80k? If yes, exp60
# becomes the new quality canonical (replaces exp52 which had 0.101 face_lpips_sq
# but no cross-attn). Speculative linear extrapolation suggests
#   exp60 face_lpips_sq portraits ~ 0.099 -- the first sub-0.10 ever.
# Even if non-linear, anything close to that resets the floor.
#
# Recipe: exp52 (80k, minimal aug, decoder_attn + pyramid + FiLM) +
# --use-cross-attn-cond. Single-flag delta vs exp52.
#
# A/B target: exp52 on val_portraits:
#   face_lpips_sq=0.101  face_lpips_vgg=0.244  face_ssim=0.579
#   whole lpips_sq=0.145  whole ssim=0.459  Δ_lpips_vgg=0.045

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp60_cross_attn_at_exp52_recipe_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp60] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp60_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp60] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp60_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp60_model_best.pt"
        echo "[exp60] resuming from $OUTDIR/exp60_model_best.pt"
    else
        echo "[exp60] no prior checkpoint, fresh start"
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
    --exp-name exp60 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp60_cross_attn_at_exp52_recipe_80k \
    --wandb-tags "exp60,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,decoder_attn,source_pyramid,film,80k,ablation_vs_exp52" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp60] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp60_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp60_final_256px_val" \
    2>&1 | tee "out/val_exp60_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp60_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp60_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp60_final_256px_val_portraits.log"

echo "[exp60] done. A/B target — exp52 (val_portraits): face_lpips_sq=0.101, face_ssim=0.579"
