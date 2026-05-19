#!/usr/bin/env bash
# exp58b — exp58 retry with --t-sample-sigma 1.5 instead of 1.0.
#
# exp58 (sigma=1.0) showed great train loss but poor val. Root cause: at
# sigma=1.0, only 0.2% of training samples land at t < 0.05 vs uniform's
# 5% — endpoints were 25x starved. Inference walks the ODE uniformly from
# t=0 to t=1, so the under-trained endpoints corrupted every trajectory
# from the first Euler step onward.
#
# Distribution at logit-normal mu=0, sigma=1.5:
#                        | t<0.05 | t<0.10 | [.4,.6] | t>0.95 |
#   uniform (baseline)   |  5.0%  | 10.0%  |  20%    |  5.0%  |
#   sigma=1.0 (exp58)    |  0.2%  |  1.4%  |  31%    |  0.2%  |  <- broken
#   sigma=1.5 (exp58b)   |  2.5%  |  7.1%  |  21%    |  2.5%  |  <- this
#   sigma=2.0            |  6.9%  | 13.6%  |  16%    |  7.1%  |  bimodal already
#
# sigma=1.5 is the narrow sweet spot: still biased toward mid-t (21% in
# [0.4,0.6] vs uniform's 20%) but endpoints only 2x starved instead of
# 25x. Going higher (sigma>=2.0) collapses the underlying Gaussian to
# bimodal — defeats the purpose.
#
# Single-flag delta vs exp58: --t-sample-sigma 1.5.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp58b_logit_normal_t_sigma15_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp58b] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp58b_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp58b] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp58b_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp58b_model_best.pt"
        echo "[exp58b] resuming from $OUTDIR/exp58b_model_best.pt"
    else
        echo "[exp58b] no prior checkpoint, fresh start"
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
    --t-sample-mode logit_normal --t-sample-mu 0.0 --t-sample-sigma 1.5 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp58b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp58b_logit_normal_t_sigma15_at_exp50_recipe \
    --wandb-tags "exp58b,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,t_logit_normal,sigma15,decoder_attn,source_pyramid,film,ablation_vs_exp50,retry_of_exp58" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp58b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp58b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp58b_final_256px_val" \
    2>&1 | tee "out/val_exp58b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp58b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp58b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp58b_final_256px_val_portraits.log"

echo "[exp58b] done. A/B target — exp50 (val_portraits): face_lpips_sq=0.124, face_ssim=0.544"
