#!/usr/bin/env bash
# exp43 — exp35 recipe (pyramid + decoder attn + minimal aug) with the flow
# off-path noise σ bumped from 0.05 → 0.30.
#
# Hypothesis: exp42 showed that removing LPIPS late produces blurry outputs —
# pure MSE on the deterministic v_target = target - source converges to a
# centroid. Adding more stochasticity to x_t during training (high σ_noise)
# means the model learns p(v | x_t, source, t) as a *distribution* rather
# than a deterministic map → less centroid-blur, more committed pattern
# choices. Inference also injects σ*noise at t=0 to match.
#
# Source-in-stem is preserved → source-conditioning signal isn't lost even
# with noisy x_t; the model has two views of source (concat'd stem + noisy x_t)
# and the noise only affects the "current point on the flow path" channel.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp43_sigma_noise_at_exp35_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --flow-sigma-noise 0.30 \
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
    --panel-keys "000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp43 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp43_sigma_noise_at_exp35_recipe \
    --wandb-tags "exp43,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,sigma_noise_03,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp43] training done. Running final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp43_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp43_final_256px" \
    2>&1 | tee "out/val_exp43_final_256px.log"

echo "[exp43] done. Compare vs:"
echo "  exp35 (σ=0.05, constant LPIPS):    face_lpips_sq=0.1526  face_ssim=0.728  lpips_vgg=0.2395"
echo "  exp42 (σ=0.05, LPIPS anneal):      face_lpips_sq=0.161   face_ssim=0.744  lpips_vgg=0.229   (blurrier visually)"
echo "  exp43 (σ=0.30, constant LPIPS):    <fresh> — testing if stochasticity breaks MSE blur"
