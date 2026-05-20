#!/usr/bin/env bash
# exp63b — PatchGAN retry with the diagnosis-fixes from exp63's silence.
#
# exp63 @ 20k vs exp61 (val_portraits) was within-noise drift on every
# metric. Visual inspection showed subtle facial-feature drift, NOT
# texture sharpening. The diagnosis was that PatchGAN was effectively
# silent for two reasons:
#
#   1. gan_weight=0.02 too weak from a pretrained G — LPIPS+flow loss
#      dominates at exp61's quality level; 0.02 adv term gets drowned.
#   2. Adaptive G/D switching starved D — from a strong pretrained G,
#      D was always "winning relative to G's adv loss", so the switch
#      kept G updating and skipped D entirely. D never developed
#      discriminative power.
#
# exp63b fixes both:
#   --gan-weight 0.05               (2.5x stronger; sweet spot for
#                                    "strong G + adversarial fine-tune")
#   --no-gan-adaptive-switch         (force D to update every step
#                                    regardless of relative loss)
#
# Same exp61 EMA resume + same 20k adversarial phase + same mid-aug
# stack.
#
# Risk: at 0.05 weight + always-on D, D may eventually dominate (the
# failure mode that killed exp21 at gan_weight=0.1). Mitigation:
# - Watch wandb d_real vs d_fake curves; if d_real - d_fake > 4 for
#   sustained periods, D is winning too hard and G can't push back.
# - Visual inspection of panels at step 5k, 10k, 15k. If panels
#   degenerate (color smearing, blockiness), kill early.
#
# A/B target — exp61 (current deployment canonical, val_portraits):
#   face_lpips_sq=0.103  face_lpips_vgg=0.242  face_ssim=0.581
#   Δ_lpips_vgg=0.025  whole lpips_sq=0.148  whole ssim=0.460
#
# Decision rule:
# - Visible texture/sharpness improvement (subjective) + metric tie
#   or small WIN → PatchGAN finally working; promote.
# - Metric LOSE (>3% face_lpips_sq) → 0.05 too strong from pretrained G;
#   try exp63c at 0.03 (middle ground).
# - Visible degradation (color smear / blockiness) → D dominated;
#   abandon PatchGAN at our scale.

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
RESUME_FROM_DEFAULT="${RESUME_FROM:-$DEFAULT_EXP61_CKPT}"

if [ ! -f "$RESUME_FROM_DEFAULT" ]; then
    echo "[exp63b] ERROR: --resume target not found at $RESUME_FROM_DEFAULT"
    echo "  Set RESUME_FROM=<path/to/exp61_model.pt> if it lives elsewhere."
    exit 1
fi

OUTDIR=out/exp63b_patchgan_stronger_from_exp61_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

# In-outdir resume takes precedence (interrupt-safe within exp63b).
LATEST_STEP_CKPT=$(ls -t "$OUTDIR"/exp63b_model_step_*.pt 2>/dev/null | head -1 || true)
if [ -n "$LATEST_STEP_CKPT" ]; then
    RESUME_TARGET="$LATEST_STEP_CKPT"
    echo "[exp63b] resuming from in-outdir ckpt: $RESUME_TARGET"
else
    RESUME_TARGET="$RESUME_FROM_DEFAULT"
    echo "[exp63b] starting from exp61 EMA: $RESUME_TARGET"
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
    --gan-weight 0.05 \
    --gan-d-channels 64 \
    --gan-d-layers 3 \
    --gan-d-lr 1e-4 \
    --gan-d-beta1 0.5 \
    --no-gan-adaptive-switch \
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
    --exp-name exp63b \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp63b_patchgan_stronger_from_exp61 \
    --wandb-tags "exp63b,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,mid_aug,cross_attn_cond,patch_gan,gan_w0.05,no_adaptive_switch,resume_from_exp61,ablation_vs_exp61" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp63b] training done. Final val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp63b_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp63b_final_256px_val" \
    2>&1 | tee "out/val_exp63b_final_256px_val.log"

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp63b_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp63b_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp63b_final_256px_val_portraits.log"

echo "[exp63b] done. A/B target — exp61 (val_portraits): face_lpips_sq=0.103  Δ_lpips_vgg=0.025"
echo "  Visual check is THE deciding metric here — texture/sharpness improvements are what GAN's for."
