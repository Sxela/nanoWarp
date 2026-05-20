#!/usr/bin/env bash
# exp64b — AdaLN-Zero retry at 80k. Tests the "needs more steps to fit
# +20% params" hypothesis raised after exp64's 20k loss.
#
# exp64 @ 20k results vs exp59 (val_portraits):
#   face_lpips_sq    0.122 -> 0.131  (+7.4% LOSE)
#   face_lpips_vgg   0.282 -> 0.300  (+6.4% LOSE)
#   delta_lpips_vgg  0.035 -> 0.048  (+37% LOSE)
#
# Caveat: +9.5M params (~20% more than exp59) at the same 20k step
# budget = ~17% fewer effective updates per param. The α gates start
# at 0 (identity-at-init) and must learn their useful configurations
# from scratch — an extra task on top of the existing conv pathway.
# Possibly under-trained at 20k.
#
# This run: same recipe as exp64, just 4x longer training (80k @ 256px
# bs=4). Cosine LR over the full 80k window. A/B target is **exp60**
# (current 80k quality canonical) not exp59 — the apples-to-apples
# 80k comparison is the only honest test.
#
# A/B target — exp60 (val_portraits):
#   face_lpips_sq  = 0.0997 (first sub-0.10 ever, current canonical)
#   face_lpips_vgg = 0.237
#   face_ssim      = 0.583
#   Δ_lpips_vgg    = 0.040
#
# Decision rule:
# - exp64b ≥ exp60 quality (face_lpips_sq ≤ 0.10): AdaLN works at our
#   scale with enough training. Promote AdaLN to canonical.
# - exp64b within 3% of exp60 but worse: lever is borderline; cite the
#   simpler `time_proj` baseline (exp60) as canonical, archive exp64b
#   as reference.
# - exp64b > 5% worse: AdaLN is dead at our scale, close the chapter.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp64b_adaln_everywhere_at_80k_noenc_attn163264_bf16_mc88_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp64b] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp64b_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp64b] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp64b_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp64b_model_best.pt"
        echo "[exp64b] resuming from $OUTDIR/exp64b_model_best.pt"
    else
        echo "[exp64b] no prior checkpoint, fresh start"
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
    --use-adaln-time \
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
    --exp-name exp64b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp64b_adaln_everywhere_at_80k \
    --wandb-tags "exp64b,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,adaln_time,decoder_attn,source_pyramid,film,80k,ablation_vs_exp60,under_train_test" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp64b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp64b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp64b_final_256px_val" \
    2>&1 | tee "out/val_exp64b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp64b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp64b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp64b_final_256px_val_portraits.log"

echo "[exp64b] done. A/B target — exp60 (val_portraits): face_lpips_sq=0.0997, face_ssim=0.583"
echo "  Verdict: exp64b ≤ 0.10 → AdaLN works with enough training; promote to canonical."
echo "           exp64b within 3% but > 0.10 → borderline; exp60 stays canonical."
echo "           exp64b > 5% worse → AdaLN is dead at our scale, chapter closed."
