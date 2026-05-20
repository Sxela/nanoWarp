#!/usr/bin/env bash
# exp66 — exp59 recipe (cross-attn @ H/8, minimal aug, flow, exp35 arch,
# 20k @ 256px bs=4) with model_ch=128 instead of 88.
#
# Tests the "is capacity the bottleneck?" question. Going 88->128 widens
# everything proportionally: c1=128, c2=256, c3=512, c4=512, cm=1024.
# Approximate param count: ~108M (~2.1x base 51M).
#
# exp22 (May 2026) tested mc=176 at 1k pairs and saw grid artifacts +
# identity collapse — too much capacity for the small dataset. With 3x
# more data now (3k mixed) and mc=128 (intermediate, not 176), the same
# failure mode shouldn't apply. But it's worth flagging as a real risk.
#
# A/B target: exp59 (mc=88, 20k):
#   val_portraits face_lpips_sq=0.122, face_lpips_vgg=0.282
# If wins on val_portraits face metrics by 3%+ at the same 20k budget,
# promote to mc=128 at 80k as exp66b vs exp60.
#
# If grid artifacts appear (visible periodicity in panels), kill the
# experiment; mc=128 is above the capacity ceiling for our data scale.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp66_wider_mc128_at_exp59_recipe_noenc_attn163264_bf16_mc128_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp66] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp66_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp66] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp66_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp66_model_best.pt"
        echo "[exp66] resuming from $OUTDIR/exp66_model_best.pt"
    else
        echo "[exp66] no prior checkpoint, fresh start"
    fi
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    $RESUME_ARG \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 128 \
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
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp66 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp66_wider_mc128_at_exp59_recipe \
    --wandb-tags "exp66,ds3k,256px,noenc,attn163264,bf16,mc128,lpips_vgg,minimal_aug,cross_attn_cond,decoder_attn,source_pyramid,film,capacity_test,ablation_vs_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp66] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp66_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp66_final_256px_val" \
    2>&1 | tee "out/val_exp66_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp66_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp66_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp66_final_256px_val_portraits.log"

echo "[exp66] done. A/B target — exp59 (val_portraits): face_lpips_sq=0.122"
