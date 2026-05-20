#!/usr/bin/env bash
# exp66b — mc=128 retry at 80k. Tests the "needs more steps to fit
# +108% params" hypothesis raised after exp66's 20k tie with exp59.
#
# exp66 @ 20k results vs exp59 (val_portraits):
#   face_lpips_sq    0.122 -> 0.126 (+3.3% slight loss, essentially TIE)
#   face_lpips_vgg   0.282 -> 0.287 (+1.8% TIE)
#   face_ssim        0.546 -> 0.543 (TIE)
#   delta_lpips_vgg  0.035 -> 0.037 (TIE)
#
# Doubling params at the same 20k budget = half the effective gradient
# updates per param. Extra capacity was sitting idle. At 80k, exp66b
# gets the same effective updates per param as exp60 (mc=88) at 80k —
# the honest capacity test.
#
# A/B target — exp60 (val_portraits, current 80k quality canonical):
#   face_lpips_sq  = 0.0997 (first sub-0.10 ever)
#   face_lpips_vgg = 0.237
#   face_ssim      = 0.583
#   Δ_lpips_vgg    = 0.040
#
# Decision rule:
# - exp66b < 0.095: mc=128 is the new canonical size, capacity was
#   the bottleneck after all
# - 0.095 ≤ exp66b ≤ 0.105: within noise of exp60, both work, mc=88
#   stays canonical (cheaper)
# - exp66b > 0.105: 50M is the right size, capacity isn't the
#   bottleneck even with 80k training
#
# Cost note: 2× params = ~2× training memory + ~1.5-2× step time.
# On Colab T4 this could be 2-3h instead of the typical 80 min.
# Verify GPU memory headroom before launching at full settings.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp66b_wider_mc128_at_80k_noenc_attn163264_bf16_mc128_256px_80k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp66b] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp66b_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp66b] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp66b_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp66b_model_best.pt"
        echo "[exp66b] resuming from $OUTDIR/exp66b_model_best.pt"
    else
        echo "[exp66b] no prior checkpoint, fresh start"
    fi
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    $RESUME_ARG \
    --steps 80000 \
    --phase1-end 0 \
    --phase2-end 80000 \
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
    --val-every 5000 --panel-every 5000 --checkpoint-every 10000 --best-every 5000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp66b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp66b_wider_mc128_at_80k \
    --wandb-tags "exp66b,ds3k,256px,noenc,attn163264,bf16,mc128,lpips_vgg,minimal_aug,cross_attn_cond,decoder_attn,source_pyramid,film,80k,ablation_vs_exp60,capacity_test,under_train_test" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp66b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp66b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp66b_final_256px_val" \
    2>&1 | tee "out/val_exp66b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp66b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp66b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp66b_final_256px_val_portraits.log"

echo "[exp66b] done. A/B target — exp60 (val_portraits): face_lpips_sq=0.0997, face_ssim=0.583"
echo "  < 0.095 → mc=128 is the new canonical size"
echo "  0.095–0.105 → within noise of mc=88; cheaper canonical wins"
echo "  > 0.105 → 50M is the right size, capacity isn't the bottleneck"
