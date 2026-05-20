#!/usr/bin/env bash
# exp65b — x0-prediction at 80k. Promotion of exp65's 20k SSIM win.
#
# exp65 @ 20k vs exp59 (val_portraits):
#   face_lpips_sq    0.122 -> 0.121  (-0.8% tie)
#   face_lpips_vgg   0.282 -> 0.269  (-4.6% WIN)
#   face_ssim        0.546 -> 0.587  (+7.5% WIN)
#   whole ssim       0.445 -> 0.559  (+25.6% WIN, huge)
#   Δ_lpips_vgg      0.035 -> 0.042  (+18.6% LOSE, robustness)
#
# Same recipe as exp65, just 4x longer training. A/B target is exp60
# (current 80k quality canonical, mc=88 + cross-attn + v-prediction).
#
# A/B target — exp60 (val_portraits):
#   face_lpips_sq  = 0.0997  face_lpips_vgg = 0.237  face_ssim = 0.583
#   whole ssim     = 0.460   Δ_lpips_vgg    = 0.040
#
# Bug-fix note (2026-05-20): in-training panels for exp65 were sampled
# with a buggy `_sample_from_source` that treated x0_hat as velocity.
# Fixed at 2026-05-20 — exp65b's in-training panels will be honest.
#
# Decision rule:
# - exp65b face_lpips_sq < 0.10 AND ssim ≥ 0.55 → x0-pred wins on
#   absolute quality + retains SSIM jump. Replaces exp60 as quality
#   canonical, and the +20% robustness regression is the deployment
#   tradeoff.
# - exp65b face_lpips_sq within 5% of exp60 AND ssim still ~0.55+ →
#   x0-pred is the SSIM canonical; exp60 stays for face_lpips_sq.
# - exp65b face_lpips_sq > 0.108 (5%+ worse) → x0-pred doesn't scale
#   to 80k; close chapter.
#
# Stretch goal: if exp65b wins, exp65c stacks x0-pred + mid-aug +
# cross-attn (exp61 + x0-prediction). Orthogonal-composition test.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp65b_x0_pred_at_80k_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp65b] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp65b_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp65b] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp65b_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp65b_model_best.pt"
        echo "[exp65b] resuming from $OUTDIR/exp65b_model_best.pt"
    else
        echo "[exp65b] no prior checkpoint, fresh start"
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
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 5000 --panel-every 5000 --checkpoint-every 10000 --best-every 5000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp65b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp65b_x0_pred_at_80k \
    --wandb-tags "exp65b,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,x0_prediction,decoder_attn,source_pyramid,film,80k,ablation_vs_exp60" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp65b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp65b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp65b_final_256px_val" \
    2>&1 | tee "out/val_exp65b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp65b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp65b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp65b_final_256px_val_portraits.log"

echo "[exp65b] done. A/B target — exp60 (val_portraits): face_lpips_sq=0.0997, ssim=0.460, face_ssim=0.583"
echo "  Look for: face_lpips_sq drops below 0.10 while keeping ssim ~0.55+ (the SSIM jump from exp65 @ 20k)."
