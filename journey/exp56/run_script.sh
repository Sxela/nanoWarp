#!/usr/bin/env bash
# exp56 — exp52 recipe (3k mixed, flow, exp35 arch, 80k @ 256px bs=4,
# LPIPS=0.2 vgg) with mid augmentation to broaden the in-distribution
# coverage for real-world inference.
#
# Motivation: exp52's "minimal aug" (scale 1.0-1.2 only, clean_prob=1.0)
# trained the model on a narrow slice of the FFHQ-aligned distribution.
# Real-world inputs vary in:
#   - head pose (tilt, pitch, yaw) — FFHQ is rigorously aligned, model
#     has zero head-pose variance
#   - face size in frame (40-100% scale)
#   - lighting / white balance
#   - mild compression / blur from phone-camera pipelines
#
# Aug stack vs prior runs:
#
#   param            | exp52 minimal | exp33b heavy (failed -16%)  | exp56 mid (this)
#   -----------------+---------------+-----------------------------+------------------
#   scale-max        | 1.2           | 1.5                          | 1.5
#   rotate-deg       | 0             | 25                           | 15
#   perspective      | 0             | 0.15 @ p=0.5                 | 0.12 @ p=0.4
#   color jitter     | 0             | 0.3 each                     | 0.15 each
#   clean-prob       | 1.0           | 0.2 (80% degraded)           | 0.7 (30% mild)
#   blur-max         | -             | 3.0                          | 1.5
#   jpeg-min         | -             | 30 (heavy)                   | 60 (mild)
#
# exp33b regressed -16% on lpips_sq vs exp23 because the heavy stack +
# 1k synth dataset couldn't absorb the aug. exp56 has 3x more data,
# 4x longer training (80k), and milder corruption-aug — every knob in
# the "destroys signal" category is cut roughly in half.
#
# A/B targets:
#   exp52 (minimal aug, val_portraits):
#     face_lpips_sq=0.101  face_lpips_vgg=0.244  face_ssim=0.579
#     whole lpips_sq=0.145  whole ssim=0.459    Δ_lpips_vgg=0.045
#
# Expected outcome:
#   - clean-val face_lpips_sq: small regression to ~0.105-0.115 (cost of aug)
#   - robustness Δ_lpips_vgg: drops toward exp32's 0.040 floor
#   - visual quality on phone-camera / tilted-head inputs: noticeably better
#
# Decision rule:
#   - clean-val regression <10%: exp56 becomes new canonical for production
#   - clean-val regression >15%: keep exp52 for benchmarks, ship exp56 only
#                                for real-input deployment

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp56_mid_aug_at_exp52_recipe_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

# Auto-resume from the most-recent checkpoint in $OUTDIR if one exists.
# Override with RESUME_FROM=/path/to/some_other.pt env var.
RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp56] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp56_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp56] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp56_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp56_model_best.pt"
        echo "[exp56] resuming from $OUTDIR/exp56_model_best.pt (no step_ ckpt yet)"
    else
        echo "[exp56] no prior checkpoint, fresh start"
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
    --exp-name exp56 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp56_mid_aug_at_exp52_recipe \
    --wandb-tags "exp56,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,mid_aug,head_pose,decoder_attn,source_pyramid,film,80k,ablation_vs_exp52" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp56] training done. Running final val on legacy val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp56_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp56_final_256px_val" \
    2>&1 | tee "out/val_exp56_final_256px_val.log"

echo "[exp56] running final val on val_portraits..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp56_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp56_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp56_final_256px_val_portraits.log"

echo "[exp56] done. Compare vs exp52 (canonical, minimal aug):"
echo "  exp52 (val_portraits): face_lpips_sq=0.101  face_lpips_vgg=0.244  face_ssim=0.579  Δ=0.045"
echo "  exp56 (val_portraits): <fresh — clean-val cost vs robustness gain>"
echo "  exp52 (legacy val):    face_lpips_sq=0.183  face_lpips_vgg=0.355  face_ssim=0.623"
echo "  exp56 (legacy val):    <fresh>"
