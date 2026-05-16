#!/usr/bin/env bash
# exp48 — pixel DiT (exp47 arch) + multiscale progressive + LPIPS warmup.
#
# Hypothesis: exp47's pixel-DiT showed block artifacts because the patch=16
# linear unpatchify produces patch-aligned outputs, and LPIPS pushes per-patch
# commitment before the model learns cross-patch coherence through attention.
#
# Fix:
#   1. Start at 128px (8×8=64 tokens at patch=16) — small enough for
#      self-attention to learn cross-patch coherence quickly with limited data.
#   2. No LPIPS during phase 1 (1k steps @ 128) — pure flow-MSE establishes
#      spatial structure and patch consistency first.
#   3. Linear LPIPS warmup from 0 → 0.2 over phase 1 (matches phase boundary).
#   4. Phase 2: 256px (16×16=256 tokens), phase 3: 512px (32×32=1024 tokens),
#      same compute-balanced shape as exp46.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp48_pixel_dit_multiscale_warmup_d384_l11_p16_prog128_256_512_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 1000 \
    --phase2-end 5000 \
    --bs-128 16 --bs-256 8 --bs-512 4 \
    --arch pixel_dit \
    --dit-pixel-dim 384 --dit-pixel-layers 11 --dit-pixel-patch 16 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-weight-warmup-steps 1000 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp48 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp48_pixel_dit_multiscale_warmup \
    --wandb-tags "exp48,ds1k,prog128_256_512,bf16,lpips_vgg,minimal_aug,pixel_dit,d384,l11,p16,lpips_warmup,ablation_vs_exp47" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp48] training done. Running final val @ 512px..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp48_model.pt" \
    --image-size 512 --batch-size 2 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp48_final_512px" \
    2>&1 | tee "out/val_exp48_final_512px.log"

echo "[exp48] also validating at 256 for apples-to-apples..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp48_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp48_final_256px" \
    2>&1 | tee "out/val_exp48_final_256px.log"

echo "[exp48] done. Compare vs:"
echo "  exp47 (single-scale 256, constant LPIPS=0.2):    block artifacts visible"
echo "  exp48 (multiscale + LPIPS warmup):               look for reduced block artifacts in face panels"
echo "  exp35 (UNet baseline, same params):              face_lpips_sq=0.1526  face_ssim=0.728"
