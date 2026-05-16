#!/usr/bin/env bash
# exp44 — combine exp32's progressive multi-res schedule, mid-strength
# corruption aug (exp33c envelope), exp35's architecture (decoder attn +
# source pyramid + FiLM), and exp42's LPIPS-weight cosine anneal 0.2 → 0.0.
#
# Goal: best-of-everything single-frame run.
#   - 100k steps progressive 128 → 256 → 512 (5k @ 128 bs=64, 20k @ 256 bs=16,
#     75k @ 512 bs=4) — proven recipe from exp32, lets the model learn
#     coarse → fine across phases.
#   - Mid-strength aug: scale [1.0, 1.5], rotate ±25°, perspective 0.15,
#     color jitter ±0.3, corruption clean_prob=0.2 with the realistic-envelope
#     tail (resize 0.5-0.9, blur ≤2.0, JPEG ≥40).
#   - exp35 arch (+ decoder attn + pyramid; 51M params).
#   - LPIPS cosine-anneal 0.2 → 0.0 over the full 100k. At step ~80k LPIPS
#     is ~0.025 — the last 20k is mostly pure MSE for detail-committing
#     convergence, riding on a model that's already on the style manifold.
#
# Final val runs at the final-phase resolution (512). Add a separate
# `--image-size 256` validate call if you want apples-to-apples vs 20k runs.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp44_progressive_mid_aug_exp35arch_lpips_anneal_100k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 100000 \
    --phase1-end 5000 \
    --phase2-end 25000 \
    --bs-128 16 --bs-256 8 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --lr 2e-4 --lr-min 1e-6 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-weight-end 0.0 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.5 \
    --aug-rotate-deg 25.0 \
    --aug-perspective 0.15 --aug-perspective-prob 0.5 \
    --aug-brightness 0.3 --aug-contrast 0.3 --aug-saturation 0.3 \
    --clean-prob 0.2 \
    --degrade-resize-prob 0.3 --degrade-resize-min 0.5 --degrade-resize-max 0.9 \
    --corrupt-blur-max 2.0 --corrupt-blur-prob 0.7 \
    --corrupt-jpeg-min 40 --corrupt-jpeg-prob 0.7 \
    --val-every 5000 --panel-every 5000 --checkpoint-every 10000 --best-every 5000 \
    --panel-keys "000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp44 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp44_progressive_mid_aug_exp35arch_lpips_anneal \
    --wandb-tags "exp44,ds1k,prog128_256_512,mc88,lpips_vgg,decoder_attn,source_pyramid,film,mid_aug,lpips_anneal,100k" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp44] training done. Running final val at 512px (training-final res)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp44_model.pt" \
    --image-size 512 --batch-size 2 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp44_final_512px" \
    2>&1 | tee "out/val_exp44_final_512px.log"

echo "[exp44] also validating at 256 for apples-to-apples vs 20k runs..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp44_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp44_final_256px" \
    2>&1 | tee "out/val_exp44_final_256px.log"

echo "[exp44] done. Compare vs reference 100k run:"
echo "  exp32 (100k progressive, base arch, full aug, no anneal) @ 512:"
echo "      lpips_sq=0.154  lpips_vgg=0.300  ssim=0.629  face_ssim=0.674  Δ=0.040"
echo "  exp44 @ 512:  <fresh>"
echo "Goal: better face quality than exp32 (exp35 arch + LPIPS anneal) while keeping Δ low."
