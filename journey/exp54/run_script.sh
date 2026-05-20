#!/usr/bin/env bash
# exp54 — re-test diffusion (eps-prediction) at the exp50 recipe.
#
# History: exp01-exp06 used classic Gaussian diffusion (eps-prediction)
# with an ImageNet-ResNet18 source encoder, default UNet, 1k synthetic
# dataset, ~2k training steps. Results were poor (loss looked OK but DDIM
# reverse-sampling collapsed to grey scribbles). exp07 switched to
# rectified flow matching and stability fixed everything; we've been flow
# ever since.
#
# Question: was flow actually the lever, or was it just the combination
# of confounders (small dataset + 2k steps + old arch + bad val signal)
# that doomed diffusion? exp54 retries diffusion with every confounder
# fixed:
#   - 3k mixed dataset (real-photo sources, not synth-only)
#   - exp35 architecture (decoder_attn + source_pyramid + FiLM)
#   - 20k steps (10x the original exp01 budget)
#   - same trainer/recipe as exp50 (so we get an honest A/B)
#   - eps-prediction (matches exp01 lineage; toggle to v via flag if needed)
#
# A/B target: exp50 (flow, same recipe) on val_portraits:
#   face_lpips_sq=0.124  face_lpips_vgg=0.285  face_ssim=0.544
#   whole lpips_sq=0.170  whole ssim=0.444
#
# Sampling budget: in-loop val (during training) uses 20 DDIM steps to
# keep iteration speed comparable with exp50 — those numbers will look
# bad for diffusion at low steps, but they're for trend-watching only.
# Final val uses 100 DDIM steps (diffusion's native sweet spot) so the
# A/B vs exp50 reflects the method's real ceiling, not its worst case.
#
# If diffusion is genuinely close to flow at this recipe, the original
# "flow is fundamentally better" thesis weakens and we'd revisit method
# choice. If it's still much worse, the gap is real.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

OUTDIR=out/exp54_diffusion_eps_at_exp50_recipe_noenc_attn163264_bf16_mc88_256px_20k
mkdir -p "$OUTDIR"

python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \
    data/photo2anime_3k \
    --method diffusion \
    --prediction-type eps \
    --diffusion-timesteps 1000 \
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
    --aug-scale-min 1.0 --aug-scale-max 1.2 \
    --aug-rotate-deg 0.0 \
    --aug-perspective 0.0 --aug-perspective-prob 0.0 \
    --aug-brightness 0.0 --aug-contrast 0.0 --aug-saturation 0.0 \
    --clean-prob 1.0 \
    --val-every 1000 --panel-every 1000 --checkpoint-every 5000 --best-every 1000 \
    --panel-keys "ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921" \
    --sample-steps 20 \
    --exp-name exp54 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp54_diffusion_eps_at_exp50_recipe \
    --wandb-tags "exp54,ds3k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,minimal_aug,decoder_attn,source_pyramid,film,diffusion,eps,ablation_vs_exp50" \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train.log"

echo "[exp54] training done. Running final val on legacy val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp54_model.pt" \
    --split val \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 100 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val \
    --outdir "out/val_exp54_final_256px_val" \
    2>&1 | tee "out/val_exp54_final_256px_val.log"

echo "[exp54] running final val on val_portraits..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_3k \
    --checkpoint "$OUTDIR/exp54_model.pt" \
    --split val_portraits \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 100 --use-ema \
    --wandb-resume "$OUTDIR" --wandb-key-prefix final_val_portraits \
    --outdir "out/val_exp54_final_256px_val_portraits" \
    2>&1 | tee "out/val_exp54_final_256px_val_portraits.log"

echo "[exp54] done. Compare vs exp50 (flow, same recipe):"
echo "  exp50 (flow,      val_portraits): face_lpips_sq=0.124  face_ssim=0.544"
echo "  exp54 (diffusion, val_portraits): <fresh — does diffusion catch up with the fixed confounders?>"
echo "  exp50 (flow,      val):           face_lpips_sq=0.201  face_ssim=0.605"
echo "  exp54 (diffusion, val):           <fresh>"
