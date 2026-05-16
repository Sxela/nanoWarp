#!/usr/bin/env bash
# exp33b — exp33 with the aug-scale range cut back to U[1.0, 1.5].
#
# Hypothesis: exp33's scale=U[1.0, 2.5] reproduced exp24b's regression
# (lpips_sq 0.168 vs exp23's 0.127), so the crop-variance dominates the aug
# stack's effect at 1k pairs / 20k steps. Conservative scale + the rest of
# the aug stack (rotate/perspective/color/blur/JPEG) should land closer to
# exp23 on clean-val while keeping the corruption-robustness benefit for
# real video.
#
# Single-arg delta vs run_exp33: `--aug-scale-max 1.5`.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp33b_aug_scale15_noenc_attn163264_bf16_mc88_256px_20k
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
    --degrade-resize-prob 0.3 --degrade-resize-min 0.25 --degrade-resize-max 0.75 \
    --corrupt-blur-max 3.0 --corrupt-blur-prob 0.7 \
    --corrupt-jpeg-min 30 --corrupt-jpeg-prob 0.7 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --sample-steps 20 \
    --exp-name exp33b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp33b_aug_scale15_at_exp23_recipe \
    --wandb-tags "exp33b,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,exp32_aug_stack,aug_scale15,ablation_vs_exp33" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp33b] training done. Running final val (25 batches, EMA, sample_steps=20)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp33b_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp33b_final_256px" \
    2>&1 | tee "out/val_exp33b_final_256px.log"

echo "[exp33b] done. Compare lpips_vgg vs exp23 (0.234) and exp33 (0.308)."
