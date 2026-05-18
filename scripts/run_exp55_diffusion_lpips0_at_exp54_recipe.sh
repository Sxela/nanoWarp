#!/usr/bin/env bash
# exp55 — diffusion (eps-prediction) at exp54 recipe with LPIPS aux turned
# off entirely.
#
# Hypothesis: LPIPS on x0_hat is harmful for diffusion at high t. Recall:
#   x0_hat = (x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)
# At high t (near pure noise), sqrt(alpha_bar) → 0, so any error in
# eps_hat gets divided by a tiny number and amplified into garbage. LPIPS
# on that garbage gives a gradient signal that pushes the model away from
# the correct eps prediction — net-negative.
#
# Flow doesn't have this pathology: x_target_hat = x_t + (1 - t) * v_hat
# is a smooth linear extrapolation that never blows up. So LPIPS works
# uniformly for flow but might be actively hurting diffusion.
#
# exp55 = exp54 recipe, single change: --lpips-weight 0.0.
# Tests pure-MSE-on-eps diffusion as the cleaner baseline.
#
# A/B targets:
#   exp50 (flow, lpips=0.2):       val_portraits face_lpips_sq=0.124
#   exp54 (diffusion, lpips=0.2):  TBD (currently running)
#   exp55 (diffusion, lpips=0.0):  <fresh — is LPIPS net-positive or net-negative for diffusion?>
#
# Decision tree:
#   exp55 > exp54: LPIPS was net-negative. Promote exp55 as canonical diffusion baseline.
#     Follow up with exp55b (lpips warmup 0→0.2 over 5k steps) to recover face quality after eps stabilizes.
#   exp55 ~ exp54: LPIPS is neutral. Drop it from diffusion recipe for simplicity.
#   exp55 < exp54: LPIPS helped despite the high-t pathology. Keep at 0.2.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp55_diffusion_eps_lpips0_at_exp54_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

# Auto-resume from the most-recent checkpoint in $OUTDIR if one exists.
# Prefers periodic step-checkpoints (latest state). Falls back to
# model_best.pt (only available between step 0 and the first --checkpoint-every
# boundary, e.g. for the user who hit the SKIP dead-loop around step 4k).
# Override with RESUME_FROM=/path/to/some_other.pt env var.
RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp55] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp55_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp55] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp55_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp55_model_best.pt"
        echo "[exp55] resuming from $OUTDIR/exp55_model_best.pt (no step_ ckpt yet)"
    else
        echo "[exp55] no prior checkpoint, fresh start"
    fi
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    $RESUME_ARG \
    --method diffusion \
    --prediction-type eps \
    --diffusion-timesteps 1000 \
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
    --lpips-weight 0.0 \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp55 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp55_diffusion_eps_lpips0_at_exp54_recipe \
    --wandb-tags "exp55,ds3k,256px,noenc,attn163264,bf16,mc88,no_lpips,minimal_aug,decoder_attn,source_pyramid,film,diffusion,eps,ablation_vs_exp54" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp55] training done. Running final val on legacy val (100 DDIM steps)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp55_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 100 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp55_final_256px_val" \
    2>&1 | tee "out/val_exp55_final_256px_val.log"

echo "[exp55] running final val on val_portraits (100 DDIM steps)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp55_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 100 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp55_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp55_final_256px_val_portraits.log"

echo "[exp55] done. A/B:"
echo "  exp50 (flow,      lpips=0.2, val_portraits): face_lpips_sq=0.124  face_ssim=0.544"
echo "  exp54 (diffusion, lpips=0.2, val_portraits): TBD"
echo "  exp55 (diffusion, lpips=0.0, val_portraits): <fresh>"
