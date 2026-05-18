#!/usr/bin/env bash
# exp53 — exp50 recipe (3k mixed, exp35 arch = decoder_attn + pyramid +
# FiLM, minimal aug, constant LPIPS=0.2, 20k @ 256px bs=4) with one
# change: PIL resize filter for the source-pool downscale is LANCZOS
# instead of BILINEAR.
#
# Motivation: FFHQ images are stored at 512px; training downscales them
# to 256 (or to 256*scale_min/max for the random zoom). BILINEAR softens
# edges noticeably on a 2x downscale. LANCZOS is the standard fix for
# preserving high-frequency detail through downscale — should give the
# model a sharper source signal to learn from, and a sharper val signal
# to be scored against.
#
# Only the *real* resize paths are switched (initial scaled zoom, val
# direct resize, post-crop fallback). Affine paths (rotate/perspective)
# stay BILINEAR — LANCZOS on sub-pixel affine introduces ringing/halos.
# Corruption-aug resize-down+up stays BILINEAR by design — it's meant
# to simulate lossy real-world inputs.
#
# A/B target: exp50 at 20k (BILINEAR) on val_portraits:
#   face_lpips_sq=0.124  face_lpips_vgg=0.285  face_ssim=0.544
#   whole lpips_sq=0.170  whole ssim=0.444

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp53_lanczos_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-resize-interp lanczos \
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp53 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp53_lanczos_at_exp50_recipe \
    --wandb-tags "exp53,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,lanczos_resize,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp53] training done. Running final val on legacy val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp53_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --aug-resize-interp lanczos \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp53_final_256px_val" \
    2>&1 | tee "out/val_exp53_final_256px_val.log"

echo "[exp53] running final val on val_portraits..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp53_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --aug-resize-interp lanczos \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp53_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp53_final_256px_val_portraits.log"

echo "[exp53] done. Compare vs exp50 (same recipe, BILINEAR):"
echo "  exp50 (20k, BILINEAR, val_portraits):  face_lpips_sq=0.124  face_lpips_vgg=0.285  face_ssim=0.544"
echo "  exp50 (20k, BILINEAR, val):            face_lpips_sq=0.201  face_lpips_vgg=0.379  face_ssim=0.605"
echo "  exp53 (20k, LANCZOS,  ...):            <fresh — does sharper source improve LPIPS/SSIM?>"
