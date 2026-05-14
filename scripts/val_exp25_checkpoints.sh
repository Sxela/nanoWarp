#!/usr/bin/env bash
set -euo pipefail
cd /home/researcher/workspace/nanoWarp

export PYTHONPATH=.:/tmp/extpkgs2
export TORCH_HOME=/tmp/torch_home
export MPLCONFIGDIR=/tmp/mpl
# WANDB_API_KEY must be set in the launching shell. Never commit keys.
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your env before running this script.}"

echo "[val_sweep] waiting for exp25 to finish..."
while [ ! -f "out/exp25_lpipsvgg_80k_from_exp23/model.pt" ]; do
    sleep 60
done
echo "[val_sweep] exp25 final model found, starting checkpoint validation sweep..."

# Validate each 10k checkpoint
for STEP in 10000 20000 30000 40000 50000 60000 70000 80000; do
    CKPT="out/exp25_lpipsvgg_80k_from_exp23/ckpt_step${STEP}.pt"
    if [ -f "$CKPT" ]; then
        echo "[val_sweep] validating step $STEP..."
        python3 experiments/010_img2img_photo2comics/validate.py \
            data/photo2anime_1k/photo2anime_1k \
            --checkpoint "$CKPT" \
            --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
            --outdir "out/val_exp25_step${STEP}" \
            2>&1 | tee "out/val_exp25_step${STEP}.log"
        echo "[val_sweep] step $STEP done."
    else
        echo "[val_sweep] checkpoint $CKPT not found, skipping."
    fi
done

# Validate final model
echo "[val_sweep] validating final model (step 80k)..."
python3 experiments/010_img2img_photo2comics/validate.py \
    data/photo2anime_1k/photo2anime_1k \
    --checkpoint out/exp25_lpipsvgg_80k_from_exp23/model.pt \
    --image-size 256 --batch-size 4 --max-batches 25 --sample-steps 20 --use-ema \
    --outdir out/val_exp25_final \
    2>&1 | tee out/val_exp25_final.log

echo "[val_sweep] all done. Printing summary..."

python3 - << 'PYEOF'
import json, os, glob

results = []
for step in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]:
    path = f"out/val_exp25_step{step}/val_metrics.json"
    if os.path.exists(path):
        d = json.load(open(path))
        results.append((step, d))

# Also final
path = "out/val_exp25_final/val_metrics.json"
if os.path.exists(path):
    d = json.load(open(path))
    results.append(("80k_final", d))

print(f"{'step':>10}  {'lpips_sq':>10}  {'lpips_vgg':>10}  {'ssim':>8}")
print("-" * 46)
for step, d in results:
    sq  = d.get("mean_lpips_squeeze_sampled", d.get("mean_lpips_sampled", 9999))
    vgg = d.get("mean_lpips_vgg_sampled", 9999)
    ssim = d.get("mean_ssim_sampled", 9999)
    print(f"{str(step):>10}  {sq:>10.4f}  {vgg:>10.4f}  {ssim:>8.4f}")
PYEOF

echo "[val_sweep] complete."
