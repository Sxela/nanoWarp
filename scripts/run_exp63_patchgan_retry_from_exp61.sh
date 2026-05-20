#!/usr/bin/env bash
# exp63 — PatchGAN adversarial loss retry, starting from exp61's EMA
# checkpoint (best deployment canonical so far).
#
# Why retry now (after exp20/21 era failed in May 2026):
# - 3x more data (3k mixed vs 1k synth) — exp21 was below GAN data floor
# - Modern arch (decoder_attn + pyramid + FiLM + cross-attn) — stronger G
# - val_portraits exists — the metric that under-counted faces in exp21
#   is gone; we can now evaluate honestly
# - Mid aug exposure (exp56) — D can't lean on input-fidelity shortcut
# - Start from exp61 EMA (80k of pretrained G) — eliminates the "G learns
#   shortcuts before basic photo->anime" failure mode entirely
# - Longer adversarial phase budget than exp21's 13k
#
# Settings rationale:
# - gan_weight=0.02: between exp21b's too-weak 0.005 and exp21's
#   too-strong 0.1. Lit-typical for image-translation PatchGAN.
# - PatchGAN ch=64, layers=3: pix2pix 70x70 default, ~2.8M params.
# - d_lr=1e-4: 2x slower than G to keep balance
# - adaptive_switch: only update the one that's currently losing more
#   (exp21c's best variant — stabilized training but val didn't improve
#    on the wrong metric)
# - NoGAN phasing dropped: starting from exp61 IS the G-pretrain, no
#   need for steps 0-5k of clean LPIPS training again.
#
# RESUME_FROM points at exp61 EMA. Step counter is reset to 1 by the
# trainer's "GAN fresh start" logic so cosine LR begins from full lr_max
# (not lr_min, which would happen if we kept exp61's step=80000).
#
# A/B target: exp61 (val_portraits, current deployment canonical):
#   face_lpips_sq=0.103  face_lpips_vgg=0.242  face_ssim=0.581
#   Δ_lpips_vgg=0.025  whole lpips_sq=0.148  whole ssim=0.460
#
# Specifically asking: does PatchGAN improve texture sharpness /
# crispness that LPIPS can't capture, at the cost of small LPIPS
# regression? exp21 reported "yes qualitatively, no quantitatively" —
# we're testing that hypothesis under a fairer measurement regime.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

# Default resume target: exp61's final EMA ckpt. Override with RESUME_FROM.
DEFAULT_EXP61_CKPT=out/exp61_cross_attn_plus_mid_aug_noenc_attn163264_bf16_mc88_256px_80k/exp61_model.pt
RESUME_FROM="${RESUME_FROM:-$DEFAULT_EXP61_CKPT}"

if [ ! -f "$RESUME_FROM" ]; then
    echo "[exp63] ERROR: --resume target not found at $RESUME_FROM"
    echo "  Set RESUME_FROM=<path/to/exp61_model.pt> if it lives elsewhere."
    exit 1
fi

OUTDIR=out/exp63_patchgan_retry_from_exp61_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

# In-outdir resume takes precedence (interrupt-safe within exp63).
LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp63_model_step_*.pt 2>/dev/null | head -1 || true)
if [ -n "$LATEST_STEP_CKPT" ]; then
    RESUME_TARGET="$LATEST_STEP_CKPT"
    echo "[exp63] resuming from in-outdir ckpt: $RESUME_TARGET"
else
    RESUME_TARGET="$RESUME_FROM"
    echo "[exp63] starting from exp61 EMA: $RESUME_TARGET"
fi

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    --resume "$RESUME_TARGET" \
    --steps 20000 \
    --phase1-end 0 \
    --phase2-end 20000 \
    --bs-128 4 --bs-256 4 --bs-512 4 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --use-decoder-attn \
    --use-source-pyramid \
    --use-cross-attn-cond \
    --use-gan \
    --gan-weight 0.02 \
    --gan-d-channels 64 \
    --gan-d-layers 3 \
    --gan-d-lr 1e-4 \
    --gan-d-beta1 0.5 \
    --gan-adaptive-switch \
    --lr 1e-4 --lr-min 1e-5 --lr-warmup-steps 500 \
    --grad-clip-norm 1.0 \
    --lpips-weight 0.2 --lpips-aux-net vgg \
    --amp bf16 \
    --aug-scale-min 1.0 --aug-scale-max 1.5 \
    --aug-rotate-deg 15.0 \
    --aug-perspective 0.12 --aug-perspective-prob 0.4 \
    --aug-brightness 0.15 --aug-contrast 0.15 --aug-saturation 0.15 \
    --clean-prob 0.7 \
    --degrade-resize-prob 0.2 --degrade-resize-min 0.5 --degrade-resize-max 0.85 \
    --corrupt-blur-max 1.5 --corrupt-blur-prob 0.5 \
    --corrupt-jpeg-min 60 --corrupt-jpeg-prob 0.5 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp63 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp63_patchgan_retry_from_exp61 \
    --wandb-tags "exp63,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,mid_aug,cross_attn_cond,patch_gan,gan_w0.02,resume_from_exp61,ablation_vs_exp61" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp63] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp63_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp63_final_256px_val" \
    2>&1 | tee "out/val_exp63_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp63_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp63_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp63_final_256px_val_portraits.log"

echo "[exp63] done. A/B target — exp61 (val_portraits): face_lpips_sq=0.103  Δ_lpips_vgg=0.025"
echo "  Also visually inspect panels for crispness / texture vs exp61."
