#!/usr/bin/env bash
# exp61 — STACK cross-attn (exp59) + mid aug (exp56) at 80k.
#
# Hypothesis: the two wins are orthogonal — exp59 is architectural
# (cross-attn enriches source->target conditioning) while exp56 is data
# (mid aug exposes the model to real-world variance). Combining should
# give both face quality + robustness simultaneously.
#
# Expected if orthogonal:
#   face_lpips_sq portraits: ~exp60's number (~0.099-0.101 speculative)
#   Δ_lpips_vgg portraits:   ~exp56's 0.027 (40% better robustness than exp52)
#
# Recipe: exp56's mid-aug stack + exp59's --use-cross-attn-cond.
# Effectively: exp52 base + mid aug + cross-attn.
#
# A/B targets:
#   exp52 (quality canonical, 80k, no aug, no cross-attn):
#     face_lpips_sq portraits=0.101  Δ_lpips_vgg=0.045
#   exp56 (deployment canonical, 80k, mid aug, no cross-attn):
#     face_lpips_sq portraits=0.104  Δ_lpips_vgg=0.027  (best robustness)
#   exp59 (architectural win, 20k, no aug, cross-attn):
#     face_lpips_sq portraits=0.122  (-1.6% vs exp50's 0.124)
#   exp61 (THIS): if orthogonal, should win on BOTH quality AND robustness.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp61_cross_attn_plus_mid_aug_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp61] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp61_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp61] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp61_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp61_model_best.pt"
        echo "[exp61] resuming from $OUTDIR/exp61_model_best.pt"
    else
        echo "[exp61] no prior checkpoint, fresh start"
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
    --exp-name exp61 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp61_cross_attn_plus_mid_aug_80k \
    --wandb-tags "exp61,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,mid_aug,head_pose,cross_attn_cond,decoder_attn,source_pyramid,film,80k,stack_of_exp56_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp61] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp61_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp61_final_256px_val" \
    2>&1 | tee "out/val_exp61_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp61_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp61_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp61_final_256px_val_portraits.log"

echo "[exp61] done. A/B targets (val_portraits):"
echo "  exp52 (canonical):       face_lpips_sq=0.101  Δ_lpips_vgg=0.045"
echo "  exp56 (deployment):      face_lpips_sq=0.104  Δ_lpips_vgg=0.027"
echo "  exp59 (cross-attn 20k):  face_lpips_sq=0.122  Δ_lpips_vgg=0.035"
echo "  exp61 (this — both):     <fresh — orthogonal stack hypothesis>"
