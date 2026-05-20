#!/usr/bin/env bash
# exp65c — STACK x0-prediction (exp65b) + mid-aug + cross-attn (exp61).
#
# Hypothesis: the two improvements are orthogonal.
# - exp65b (x0-prediction, exp59 recipe + x0): wins on face_lpips_sq
#   tie / face_lpips_vgg -5% / face_ssim +9% / whole_ssim +29% /
#   legacy face_lpips_sq -10%. The one cost: Δ_lpips_vgg +17%.
# - exp61 (mid aug + cross-attn): best Δ_lpips_vgg=0.025 ever measured,
#   via training-time corruption exposure (clean_prob=0.7 + mild blur
#   + jpeg + scale/rotate/perspective/color jitter).
#
# Composition prediction: x0-prediction's optimization-target benefit
# is independent of mid-aug's data-distribution benefit. If they stack
# cleanly (like exp59 cross-attn + exp56 mid-aug → exp61 did), the
# result is best face_lpips_sq AND best Δ_lpips_vgg in one model.
#
# Recipe: exp61 + --flow-prediction-type x0. Single-flag delta.
#
# A/B targets (val_portraits):
#   exp65b (new quality canonical):
#     face_lpips_sq=0.0996  face_lpips_vgg=0.226  face_ssim=0.635
#     whole ssim=0.593       Δ_lpips_vgg=0.047
#   exp61 (deployment canonical):
#     face_lpips_sq=0.103   face_lpips_vgg=0.242  face_ssim=0.581
#     whole ssim=0.460       Δ_lpips_vgg=0.025
#   exp65c target if orthogonal:
#     face_lpips_sq~0.10    whole ssim~0.59       Δ_lpips_vgg~0.030
#
# Decision rule:
# - face_lpips_sq ≤ 0.103 AND Δ_lpips_vgg ≤ 0.032 → NEW SINGLE
#   CANONICAL (replaces both exp65b and exp61). Best quality + best
#   robustness in one model.
# - face_lpips_sq ≤ 0.103 AND Δ_lpips_vgg in (0.032, 0.040] → quality
#   canonical with reasonable robustness; replaces exp65b. exp61
#   stays as the pure-robustness reference.
# - face_lpips_sq > 0.105 → mid-aug interferes with x0-pred's
#   optimization target. Keep them separate: exp65b for quality,
#   exp61 for deployment.
# - Δ_lpips_vgg > 0.040 → mid-aug didn't recover the x0-pred
#   robustness regression. Same: keep them separate.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp65c_x0_pred_plus_mid_aug_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp65c] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp65c_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp65c] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp65c_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp65c_model_best.pt"
        echo "[exp65c] resuming from $OUTDIR/exp65c_model_best.pt"
    else
        echo "[exp65c] no prior checkpoint, fresh start"
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
    --flow-prediction-type x0 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.5 \
    --aug-rotate-deg 15.0 \
    --aug-perspective 0.12 --aug-perspective-prob 0.4 \
    --aug-brightness 0.15 --aug-contrast 0.15 --aug-saturation 0.15 \
    --clean-prob 0.7 \
    --degrade-resize-prob 0.2 --degrade-resize-min 0.5 --degrade-resize-max 0.85 \
    --corrupt-blur-max 1.5 --corrupt-blur-prob 0.5 \
    --corrupt-jpeg-min 60 --corrupt-jpeg-prob 0.5 \
    --val-every 5000 --panel-every 5000 --checkpoint-every 10000 --best-every 5000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp65c \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp65c_x0_pred_plus_mid_aug_80k \
    --wandb-tags "exp65c,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,mid_aug,cross_attn_cond,x0_prediction,decoder_attn,source_pyramid,film,80k,stack_of_exp65b_exp61" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp65c] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp65c_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp65c_final_256px_val" \
    2>&1 | tee "out/val_exp65c_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp65c_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp65c_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp65c_final_256px_val_portraits.log"

echo "[exp65c] done. A/B targets (val_portraits):"
echo "  exp65b (quality canonical):    face_lpips_sq=0.0996  whole ssim=0.593  Δ=0.047"
echo "  exp61  (deployment canonical): face_lpips_sq=0.103   whole ssim=0.460  Δ=0.025"
echo "  exp65c (this — stack target):  face_lpips_sq~0.10   whole ssim~0.59   Δ~0.030"
echo "  If face_lpips_sq ≤ 0.103 AND Δ ≤ 0.032 → NEW SINGLE CANONICAL."
