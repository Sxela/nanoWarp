#!/usr/bin/env bash
# exp36 — exp35 recipe (minimal aug + decoder attn + pyramid) + DiT bottleneck.
#
# Architecture delta vs exp35: the conv bottleneck (mid_attn + mid2 ResBlock)
# is replaced by 4 DiT-XL-style transformer blocks operating on the flattened
# (H/16 × W/16, cm=704) token grid. `mid1` still projects c4 → cm upstream
# so the DiT stack operates at constant width.
#
# adaLN-zero init → identity at step 0; safe to stack on top of exp35's
# architecture. Adds ~28M params on top of exp35 (~51M → ~79M total).
# No inference-time external dependencies.
#
# No contrastive loss here — clean architecture-only A/B vs exp35. Contrastive
# can be layered later if DiT shows promise.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp36_dit_at_exp35_recipe_noenc_attn163264_bf16_mc88_256px_20k
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
    --use-dit-bottleneck \
    --num-dit-blocks 4 \
    --dit-mlp-ratio 4.0 \
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
    --sample-steps 20 \
    --exp-name exp36 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp36_dit_at_exp35_recipe \
    --wandb-tags "exp36,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,dit_bottleneck,ablation_vs_exp35" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp36] training done. Running final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp36_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp36_final_256px" \
    2>&1 | tee "out/val_exp36_final_256px.log"

echo "[exp36] done. Headline comparisons:"
echo "  exp35 (no DiT):  face_lpips_sq=0.1526  face_lpips_vgg=0.2859  lpips_vgg=0.2395  params=51M"
echo "  exp36 (DiT):     <fresh>  expected params=79M (+28M)"

