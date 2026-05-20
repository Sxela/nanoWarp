#!/usr/bin/env bash
# exp64 — exp59 recipe (cross-attn @ H/8, minimal aug, exp35 arch,
# 20k @ 256px bs=4) with AdaLN-Zero time conditioning in every ResBlock.
#
# Architectural shift: replaces every ResBlock with AdaLNResBlock. The
# additive `h = h + time_proj(t_emb)` injection is gone. Each ResBlock
# now gets full DiT/SD3-style modulation:
#   norm1 -> γ1·x + β1 -> conv1 -> α1·h -> norm2 -> γ2·x + β2 -> conv2 -> α2·h -> + skip
# where (γ,β,α) come from a per-block linear projection of t_emb.
# Output gates α are zero-init so the block is identity at insertion
# time; safe vs the baseline trajectory.
#
# Param cost: +9.5M (~20%) vs exp59 — the per-block modulation MLPs add
# up across 10 ResBlocks. Total ~58M.
#
# Single-flag delta vs exp59: --use-adaln-time.
#
# A/B target: exp59 (val_portraits): face_lpips_sq=0.122
#
# Modern flow/diffusion lit (DiT, SD3, Flux) consistently reports
# 1-3% gains from AdaLN-Zero modulation. Our scale (~50M) is lower
# than DiT-XL (~675M), so the gain may be smaller. If clean win at 20k,
# promote to 80k as exp64b vs exp60.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp64_adaln_everywhere_at_exp59_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
    echo "[exp64] resuming from RESUME_FROM=${RESUME_FROM}"
else
    LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp64_model_step_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_STEP_CKPT" ]; then
        RESUME_ARG="--resume ${LATEST_STEP_CKPT}"
        echo "[exp64] resuming from ${LATEST_STEP_CKPT}"
    elif [ -f "$OUTDIR/exp64_model_best.pt" ]; then
        RESUME_ARG="--resume $OUTDIR/exp64_model_best.pt"
        echo "[exp64] resuming from $OUTDIR/exp64_model_best.pt"
    else
        echo "[exp64] no prior checkpoint, fresh start"
    fi
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    $RESUME_ARG \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --use-cross-attn-cond \
    --use-adaln-time \
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
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp64 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp64_adaln_everywhere_at_exp59_recipe \
    --wandb-tags "exp64,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,cross_attn_cond,adaln_time,decoder_attn,source_pyramid,film,ablation_vs_exp59" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp64] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp64_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp64_final_256px_val" \
    2>&1 | tee "out/val_exp64_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp64_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp64_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp64_final_256px_val_portraits.log"

echo "[exp64] done. A/B target — exp59 (val_portraits): face_lpips_sq=0.122"
