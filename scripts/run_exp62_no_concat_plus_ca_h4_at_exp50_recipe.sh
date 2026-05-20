#!/usr/bin/env bash
# exp62 — drop source-in-stem + add cross-attn at H/4.
#
# Hypothesis: source is currently fed to the model THREE ways:
#   (1) source-in-stem: concatenated into in_conv (channels 0-2)
#   (2) source pyramid: separate conv stack -> features at 4 resolutions
#   (3) FiLM + cross-attn @ H/8: pyramid features modulate decoder
#
# With (2) and (3) both in place (since exp35/exp59), (1) may be
# redundant. In flow matching, x_t = (1-t)*source + t*target, so at t=0
# the model sees source via x_t itself — even without source-in-stem.
# At mid-t, source comes from the pyramid pathway.
#
# exp62 tests removing (1) AND adding cross-attn at H/4 to compensate
# (multi-scale cross-attn: H/8 [1024 tokens] + H/4 [4096 tokens]).
# Net param delta vs exp60: +495k (essentially same budget).
#
# Code changes:
# - --no-source-in-stem flag: in_conv goes 6->88 to 3->88, encoder sees
#   only noisy_target.
# - --use-cross-attn-cond-h4 flag: adds CrossAttnCond at H/4 decoder
#   level (target_ch=c3, cond_ch=c3, +500k params).
# - model.py: relaxed the "use_source_encoder=False forces
#   source_in_stem=True" override when pyramid is enabled.
#
# A/B target: exp59 (20k, cross-attn @ H/8 only, source_in_stem=True):
#   face_lpips_sq portraits = 0.122
#   face_lpips_vgg portraits = 0.282
#   Δ_lpips_vgg portraits   = 0.035
#
# 20k for cheap A/B. If wins, promote to 80k as exp62b vs exp60 (the
# new quality canonical at 0.0997).

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp62_no_concat_plus_ca_h4_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp62] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp62_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp62] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp62_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp62_model_best.pt"
        echo "[exp62] resuming from $OUTDIR/exp62_model_best.pt"
    else
        echo "[exp62] no prior checkpoint, fresh start"
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
    --use-cross-attn-cond \
    --use-cross-attn-cond-h4 \
    --no-source-in-stem \
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
    --exp-name exp62 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp62_no_concat_plus_ca_h4_at_exp50_recipe \
    --wandb-tags "exp62,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,no_source_in_stem,cross_attn_cond,cross_attn_h4,decoder_attn,source_pyramid,film,ablation_vs_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp62] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp62_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp62_final_256px_val" \
    2>&1 | tee "out/val_exp62_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp62_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp62_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp62_final_256px_val_portraits.log"

echo "[exp62] done. A/B target — exp59 (val_portraits): face_lpips_sq=0.122"
