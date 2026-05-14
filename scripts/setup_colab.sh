#!/usr/bin/env bash
# Colab bootstrap for nanoWarp.
#
# Usage (in a Colab cell):
#   !git clone https://github.com/<owner>/nanoWarp.git /content/nanoWarp
#   %cd /content/nanoWarp
#   !bash scripts/setup_colab.sh
#
# Then to launch exp33:
#   import os; os.environ["WANDB_API_KEY"] = "wandb_v1_..."   # in a Python cell, or:
#   !WANDB_API_KEY=wandb_v1_... bash scripts/run_exp33_aug32stack_at_exp23_recipe.sh
#
# NEVER paste the key into a notebook cell that gets committed to git.

set -euo pipefail

echo "[colab] python: $(python3 --version)"
echo "[colab] gpu:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
    echo "  (no GPU detected — switch to a GPU runtime via Runtime > Change runtime type)"

# ---------------------------------------------------------------------------
# Python dependencies
# Colab pre-installs torch / torchvision / numpy / PIL / matplotlib. We only
# add the project-specific extras and pin nothing aggressively to avoid
# clobbering the pre-built torch+CUDA combo.
# ---------------------------------------------------------------------------

echo "[colab] installing extra deps..."
pip install --quiet \
    wandb \
    lpips \
    torchmetrics \
    av \
    pyyaml \
    opencv-python

# ---------------------------------------------------------------------------
# Data layout
#
# The exp33 training command expects data at:
#   data/photo2anime_1k/photo2anime_1k/{train,val}/{source,target}/
#
# Easiest options on Colab:
#   (a) Mount Drive and symlink:
#         from google.colab import drive
#         drive.mount("/content/drive")
#         !ln -sf /content/drive/MyDrive/datasets/photo2anime_1k data/photo2anime_1k
#   (b) Copy from Drive (faster IO than reading directly from Drive):
#         !cp -r /content/drive/MyDrive/datasets/photo2anime_1k data/
#   (c) Download from a URL with wget / gdown — only if hosted somewhere public.
#
# Verify the layout before launching:
# ---------------------------------------------------------------------------

if [ ! -d "data/photo2anime_1k/photo2anime_1k/train/source" ]; then
    echo "[colab][WARN] data/photo2anime_1k/photo2anime_1k/train/source not found."
    echo "             Place the dataset under data/ before launching training."
    echo "             See the comment block above for Drive-mount and copy options."
else
    n_train=$(find "data/photo2anime_1k/photo2anime_1k/train/source" -maxdepth 1 -type f | wc -l)
    n_val=$(find "data/photo2anime_1k/photo2anime_1k/val/source" -maxdepth 1 -type f 2>/dev/null | wc -l)
    echo "[colab] dataset OK: train=$n_train pairs  val=$n_val pairs"
fi

# ---------------------------------------------------------------------------
# Runtime env (the run scripts already export defaults for these, but setting
# them here keeps them consistent across multiple training launches in the
# same Colab session).
# ---------------------------------------------------------------------------

export PYTHONPATH="."
export TORCH_HOME="/tmp/torch_home"
export MPLCONFIGDIR="/tmp/mpl"
export WANDB_CACHE_DIR="/tmp/wandb_cache"
export WANDB_CONFIG_DIR="/tmp/wandb_config"
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Verify the project import path is wired before the training script tries it.
python3 -c "from src.img2img import Img2ImgDiffusionUNet; print('[colab] src.img2img import OK')"

# Quick torch / CUDA sanity check — bf16 needs SM >= 8.0 (A100, L4, A10).
# T4 (SM 7.5) and V100 (SM 7.0) fall back to fp32 silently in bf16 mode, which
# is correct numerically but ~2x slower. Switch to --amp no on those GPUs, or
# accept the slowdown.
python3 - <<'PY'
import torch
print(f"[colab] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = "native bf16 OK" if cap[0] >= 8 else "no native bf16 — use --amp no for speed"
    print(f"[colab]   gpu={name}  sm={cap[0]}.{cap[1]}  vram={vram:.1f}GB  ({bf16})")
PY

echo "[colab] setup done. Launch with:"
echo "  WANDB_API_KEY=wandb_v1_... bash scripts/run_exp33_aug32stack_at_exp23_recipe.sh"
