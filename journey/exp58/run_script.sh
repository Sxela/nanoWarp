#!/usr/bin/env bash
# exp58 — exp50 recipe with logit-normal t-sampling instead of uniform.
#
# Default flow training samples t ~ U[0,1]. Endpoints (t=0 = source,
# t=1 = target) are "easy" — the model just learns the full delta. The
# hard region is mid-t where x_t is a partial interpolant and the model
# has to predict velocity from a mixed signal.
#
# SD3 / EDM family use logit-normal: t = sigmoid(N(mu, sigma)). With
# mu=0, sigma=1 it peaks at 0.5 (full weight on the hard middle of the
# path) while still covering the tails. Reported gains: better sample
# quality at fewer steps, especially for high-fidelity tasks.
#
# Single-flag delta vs exp50: --t-sample-mode logit_normal --t-sample-mu 0 --t-sample-sigma 1.
#
# A/B target: exp50 (uniform t) on val_portraits:
#   face_lpips_sq=0.124, face_lpips_vgg=0.285, face_ssim=0.544
# 20k for cheap A/B; if it wins, promote to 80k vs exp52.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp58_logit_normal_t_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp58] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp58_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp58] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp58_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp58_model_best.pt"
        echo "[exp58] resuming from $OUTDIR/exp58_model_best.pt"
    else
        echo "[exp58] no prior checkpoint, fresh start"
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
    --t-sample-mode logit_normal --t-sample-mu 0.0 --t-sample-sigma 1.0 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp58 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp58_logit_normal_t_at_exp50_recipe \
    --wandb-tags "exp58,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,t_logit_normal,decoder_attn,source_pyramid,film,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp58] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp58_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp58_final_256px_val" \
    2>&1 | tee "out/val_exp58_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp58_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp58_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp58_final_256px_val_portraits.log"

echo "[exp58] done. A/B target — exp50 (val_portraits): face_lpips_sq=0.124, face_ssim=0.544"
