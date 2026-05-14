#!/usr/bin/env bash
set -euo pipefail
cd /home/researcher/workspace/nanoWarp

export PYTHONPATH=.:/tmp/extpkgs2
export TORCH_HOME=/tmp/torch_home
export MPLCONFIGDIR=/tmp/mpl
# WANDB_API_KEY must be set in the launching shell. Never commit keys.
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

echo "[pipeline] waiting for exp24 to finish..."
while [ ! -f "out/exp24_lpipsvgg_nozoom_noenc_attn163264_bf16_mc88_256px_20k/model.pt" ]; do
    sleep 60
done
echo "[pipeline] exp24 done, starting exp24b..."

python3 experiments/010_img2img_photo2comics/train.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 20000 \
    --image-size 256 \
    --batch-size 4 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 --lr-cosine \
    --grad-clip-norm 1.0 \
    --no-source-encoder \
    --source-dropout 0.0 \
    --method flow --flow-sigma-noise 0.05 \
    --amp bf16 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --lpips-weight 0.2 \
    --lpips-aux-net vgg \
    --aug-resize-scale 2.0 \
    --aug-scale-jitter 0.0 \
    --sample-panel-steps 20 \
    --checkpoint-every 5000 \
    --val-every 1000 \
    --panel-every 1000 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k \
    --wandb-tags "exp24b,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,scale2" \
    --outdir out/exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k \
    2>&1 | tee out/exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k.log

echo "[pipeline] exp24b done. Running val..."

python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint out/exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k/model.pt \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir out/val_exp24b_final_256px \
    2>&1 | tee out/val_exp24b_final_256px.log

# Pick winner: exp23 vs exp24b by lpips_vgg (honest metric)
EXP23_VGG=$(python3 -c "import json; d=json.load(open('out/val_exp23_final_256px/val_metrics.json')); print(d.get('mean_lpips_vgg_sampled', 9999))")
EXP24B_VGG=$(python3 -c "import json; d=json.load(open('out/val_exp24b_final_256px/val_metrics.json')); print(d.get('mean_lpips_vgg_sampled', 9999))")

echo "[pipeline] exp23 lpips_vgg=$EXP23_VGG  exp24b lpips_vgg=$EXP24B_VGG"

WINNER_CKPT=$(python3 -c "
a, b = $EXP23_VGG, $EXP24B_VGG
if a <= b:
    print('out/exp23_lpips_vgg_noenc_attn163264_bf16_mc88_256px_20k/model.pt')
else:
    print('out/exp24b_lpipsvgg_scale2_noenc_attn163264_bf16_mc88_256px_20k/model.pt')
")
WINNER_NAME=$(python3 -c "
a, b = $EXP23_VGG, $EXP24B_VGG
print('exp23' if a <= b else 'exp24b')
")

echo "[pipeline] winner: $WINNER_NAME ($WINNER_CKPT) — launching 80k run..."

python3 experiments/010_img2img_photo2comics/train.py \
    data/photo2anime_1k/photo2anime_1k \
    --steps 80000 \
    --image-size 256 \
    --batch-size 4 \
    --lr 2e-4 --lr-min 1e-5 --lr-warmup-steps 500 --lr-cosine \
    --grad-clip-norm 1.0 \
    --no-source-encoder \
    --source-dropout 0.0 \
    --method flow --flow-sigma-noise 0.05 \
    --amp bf16 \
    --model-ch 88 \
    --attn-resolutions "16,32,64" \
    --lpips-weight 0.2 \
    --lpips-aux-net vgg \
    --aug-resize-scale $(python3 -c "print('2.0' if '$WINNER_NAME' == 'exp24b' else '1.10')") \
    --aug-scale-jitter 0.0 \
    --sample-panel-steps 20 \
    --checkpoint-every 10000 \
    --val-every 5000 \
    --panel-every 5000 \
    --wandb \
    --wandb-project nanoWarp \
    --wandb-run-name exp25_best_lpipsvgg_80k_from_${WINNER_NAME} \
    --wandb-tags "exp25,ds1k,256px,noenc,attn163264,bf16,mc88,lpips_vgg,80k,from_${WINNER_NAME}" \
    --outdir out/exp25_best_lpipsvgg_80k_from_${WINNER_NAME} \
    2>&1 | tee out/exp25_best_lpipsvgg_80k_from_${WINNER_NAME}.log

echo "[pipeline] exp25 80k done."
