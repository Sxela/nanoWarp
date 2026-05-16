#!/usr/bin/env bash
# exp33c — exp33b's scale-1.5 recipe + corruption tail dialled back to a
# realistic web-video envelope: only blur + JPEG, no resize-down-up.
#
# Rationale: real video frames at their target resolution don't go through
# a downscale-then-upscale cycle — that artifact only occurs for thumbnails
# blown up to display size, which isn't the deployment case. JPEG/codec
# blockiness and mild defocus / motion blur are the actual compression
# artifacts the model will see. exp33 / exp33b's clean-val regression vs
# exp23 (0.308 / 0.274 vs 0.234) is partly the rare extreme tail of the
# corruption distribution burning training capacity on inputs that never
# occur in real footage.
#
# Aug recipe deltas vs exp33b:
#   --degrade-resize-min  0.25 → 0.5   (when resize fires: 4× area max, not 16×)
#   --degrade-resize-max  0.75 → 0.9   (most resize cases very mild)
#   --corrupt-blur-max    3.0  → 2.0   (out-of-focus extreme cut)
#   --corrupt-jpeg-min    30   → 40    (skip worst block artifacts)
#
# Everything else identical to exp33b (scale [1.0, 1.5], rotate ±25°,
# perspective 0.15, color jitter ±0.3, hflip, exp23-style architecture).
#
# Expected outcome: clean lpips_vgg between exp33b's 0.274 and exp23's 0.234;
# corruption-val Δ smaller than exp25's 0.116 but maybe larger than exp32's
# 0.064 (less extreme training = less robust against very-bad inputs).

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp33c_milder_corruption_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.5 \
    --aug-rotate-deg 25.0 \
    --aug-perspective 0.15 --aug-perspective-prob 0.5 \
    --aug-brightness 0.3 --aug-contrast 0.3 --aug-saturation 0.3 \
    --clean-prob 0.2 \
    --degrade-resize-prob 0.3 --degrade-resize-min 0.5 --degrade-resize-max 0.9 \
    --corrupt-blur-max 2.0 --corrupt-blur-prob 0.7 \
    --corrupt-jpeg-min 40 --corrupt-jpeg-prob 0.7 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --sample-steps 20 \
    --exp-name exp33c \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp33c_milder_corruption_at_exp23_recipe \
    --wandb-tags "exp33c,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,exp23_aug,mild_corruption,ablation_vs_exp33b" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp33c] training done. Running final val (25 batches, EMA, sample_steps=20) — includes corruption-val Δ..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp33c_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp33c_final_256px" \
    2>&1 | tee "out/val_exp33c_final_256px.log"

echo "[exp33c] done. Compare on the lpips_vgg axis:"
echo "  exp23  (no aug):                              0.234   (clean)"
echo "  exp33b (scale 1.5 + full corruption):         0.274   (clean)  Δ=?"
echo "  exp33c (scale 1.5 + MILDER corruption):       <fresh>          Δ=<fresh>"
echo "  exp32  (scale 2.5 + full corruption):         0.265   (clean)  Δ=0.064"
echo "  exp25  (no aug, 20k):                         0.234   (clean)  Δ=0.116"
echo "Goal: exp33c clean closer to exp23 than exp33b was, Δ still meaningfully below 0.116."
