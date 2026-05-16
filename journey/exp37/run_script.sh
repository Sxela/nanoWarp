#!/usr/bin/env bash
# exp37 — exp23-equivalent recipe (minimal aug, no corruption) + symmetric
# decoder spatial self-attention.
#
# Two goals in one run:
#   1. Architecture A/B: does symmetric decoder attn help when the rest of
#      the recipe is the proven exp23 baseline? exp34 stacks attn on top of
#      heavy aug (which itself regressed clean-val); exp37 stacks attn on
#      top of light aug to isolate the architecture effect cleanly.
#   2. Corruption-Δ reference: lets us measure a clean-trained model's
#      robustness-gap *with the new metric*, giving us a third anchor (next
#      to exp25 at Δlpips_vgg≈0.116 and exp32 at Δ≈0.064). The lost exp25
#      checkpoint motivated this — we needed a clean-trained reference at
#      20k steps to put on the Δ axis next to exp32-style training.
#
# Aug settings ≈ exp23 (the new train_exp32_prog512.py reproduces exp23-style
# behaviour when geometric/color aug is dialled to ~identity and corruption
# is fully skipped via clean_prob=1.0):
#   - scale ∈ [1.0, 1.2]   (matches exp23's resize_scale=1.10 + jitter 0.10)
#   - rotate ±0°, perspective off, color jitter off
#   - clean_prob = 1.0 → degradation pipeline is completely skipped
#   - hflip stays at 0.5 (matches exp23, no CLI knob)

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp37_decoder_attn_at_exp23_recipe_noenc_attn163264_bf16_mc88_256px_20k
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
    --exp-name exp37 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp37_decoder_attn_at_exp23_recipe \
    --wandb-tags "exp37,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,ablation_vs_exp23" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp37] training done. Running final val (25 batches, EMA, sample_steps=20)..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint "$OUTDIR/exp37_model.pt" \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir "out/val_exp37_final_256px" \
    2>&1 | tee "out/val_exp37_final_256px.log"

echo "[exp37] done. Compare vs:"
echo "  exp23 step 20k (clean, no attn):       lpips_vgg=0.234  Δ=? (would need re-val)"
echo "  exp25 step 20k (clean, no attn):       lpips_vgg=0.234  Δ=0.116"
echo "  exp32 step 20k (full aug, no attn):    lpips_vgg=0.265  Δ=0.064"
echo "  exp37        (minimal aug + attn):    fresh data point"
