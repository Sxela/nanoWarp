#!/usr/bin/env bash
# exp47 — pure-pixel DiT (HiDream-O1 style) at 49M params, same recipe slot
# as exp35: minimal aug, 20k steps @ 256px, bs=4.
#
# Architecture: PixelDiT (src/img2img/dit_pixel.py).
#   - Patch=16 → 16×16 = 256 tokens @ 256px (1024 @ 512px later).
#   - dim=384, num_layers=11, num_heads=6 (head_dim=64) → 48.5M params.
#   - Source concatenated channel-wise with noisy_target before patchify
#     (source_in_stem=True): 6-channel patches.
#   - adaLN-zero conditioning on t_emb per block; zero-init final head.
#   - 2D sinusoidal positional embeddings (size-agnostic).
#
# This is a clean A/B against exp35's UNet at the same param budget,
# same data, same minimal aug, same LPIPS-VGG loss. Tests whether
# the HiDream-style pixel-DiT approach is competitive at small scale.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp47_pixel_dit_at_exp35_recipe_d384_l11_p16_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --arch pixel_dit \
    --dit-pixel-dim 384 --dit-pixel-layers 11 --dit-pixel-patch 16 \
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
    --exp-name exp47 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp47_pixel_dit_at_exp35_recipe \
    --wandb-tags "exp47,ds1k,256px,bf16,lpips_vgg,minimal_aug,pixel_dit,d384,l11,p16,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp47] training done. Running final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp47_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp47_final_256px" \
    2>&1 | tee "out/val_exp47_final_256px.log"

echo "[exp47] done. Compare vs exp35 (same recipe, UNet arch):"
echo "  exp35:  face_lpips_sq=0.1526  face_lpips_vgg=0.2859  face_ssim=0.728  lpips_vgg=0.2395  (49M UNet)"
echo "  exp47:  <fresh>                                                                          (48.5M PixelDiT)"
