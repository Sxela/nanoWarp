"""exp32 — train from scratch with progressive resolution + randomized corruptions.

Progressive resolution schedule:
  5k  steps @ 128px  bs=64  — structure bootstrap
  20k steps @ 256px  bs=16  — detail learning
  75k steps @ 512px  bs=4   — high-res quality
  Total: 100k steps

Per-sample randomized augmentations:
  resize scale  ~ U[scale_min, scale_max]  (default 1.0–2.5)
  blur sigma    ~ U[0.5, blur_max]          applied with p=blur_prob
  JPEG quality  ~ U[jpeg_min, 95]           applied with p=jpeg_prob
  clean pass    prob = clean_prob (20%)

Architecture: mc=88, no source encoder (source in stem), attn_res=(16,32,64),
              flow FM, LPIPS-VGG weight 0.2.  image_size=512 throughout.

Usage:
    OUTDIR=out/exp32_prog512_$(date +%Y%m%d_%H%M%S)
    mkdir -p $OUTDIR
    PYTHONPATH=/tmp/extpkgs2:/home/researcher/workspace/nanoWarp \\
    TORCH_HOME=/tmp/torch_home \\
    WANDB_API_KEY=wandb_v1_... \\
    WANDB_CACHE_DIR=/tmp/wandb_cache \\
    WANDB_CONFIG_DIR=/tmp/wandb_config \\
    MPLCONFIGDIR=/tmp/mplconfig \\
    python3 experiments/010_img2img_photo2comics/train_exp32_prog512.py \\
        data/photo2anime_1k/photo2anime_1k \\
        --wandb --wandb-run-name exp32_prog512 \\
        --outdir $OUTDIR \\
        2>&1 | tee $OUTDIR/train.log
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import InterpolationMode

from src.img2img import EMA, Img2ImgDiffusionUNet, PixelDiT, resolve_resize_interp
from src.img2img.flow import FlowConfig, RectifiedImageFlow
from src.img2img.diffusion import DiffusionConfig, GaussianImageDiffusion
from src.img2img.render import save_val_panel
from src.utils.config import apply_yaml_config


# ---------------------------------------------------------------------------
# Phase schedule: (end_step, image_size, batch_size)  — overridden in main()
# ---------------------------------------------------------------------------

PHASES: list[tuple[int, int, int]] = [
    (5_000,   128, 64),
    (25_000,  256, 16),
    (100_000, 512,  4),
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_pairs(root: Path, split: str | None) -> list[tuple[Path, Path]]:
    base = root / split if split and (root / split).exists() else root
    src_dir = base / "source"
    tgt_dir = base / "target"
    src = {p.stem: p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in IMG_EXTS}
    tgt = {p.stem: p for p in sorted(tgt_dir.iterdir()) if p.suffix.lower() in IMG_EXTS}
    keys = sorted(set(src) & set(tgt))
    if not keys:
        raise ValueError(f"No paired images found in {src_dir} / {tgt_dir}")
    return [(src[k], tgt[k]) for k in keys]


def _perspective_params(w: int, h: int, distortion: float):
    """Random perspective warp params.  Same for source and target."""
    dw, dh = distortion * w, distortion * h
    sp = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    ep = [
        [int(random.uniform(0, dw)),          int(random.uniform(0, dh))],
        [int(random.uniform(w - dw, w - 1)),  int(random.uniform(0, dh))],
        [int(random.uniform(w - dw, w - 1)),  int(random.uniform(h - dh, h - 1))],
        [int(random.uniform(0, dw)),           int(random.uniform(h - dh, h - 1))],
    ]
    return sp, ep


def _jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf)).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)


class ProgPairedDataset(Dataset):
    """Paired dataset with full augmentation pipeline.

    Shared geometry (source + target):
      random zoom scale ~ U[scale_min, scale_max] → resize → random crop
      random rotation ± rotate_deg
      random perspective warp (prob = perspective_prob)
      random horizontal flip

    Source-only color jitter (PIL, before to_tensor):
      brightness, contrast, saturation — always applied with small random factors

    Source-only degradation (tensor, gated by clean_prob):
      resize-down+up  (internet/interlaced style, prob = resize_degrade_prob)
      Gaussian blur   (prob = blur_prob)
      JPEG compression (prob = jpeg_prob)

    Validation (val=True): resize to image_size, no augmentation, no corruption.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        image_size: int,
        # geometry
        scale_min: float = 1.0,
        scale_max: float = 2.5,
        rotate_deg: float = 10.0,
        perspective_distortion: float = 0.15,
        perspective_prob: float = 0.5,
        hflip_prob: float = 0.5,
        # source-only color jitter
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        # source-only degradation
        clean_prob: float = 0.2,
        resize_degrade_prob: float = 0.3,
        resize_degrade_min: float = 0.25,
        resize_degrade_max: float = 0.75,
        blur_max: float = 3.0,
        blur_prob: float = 0.7,
        jpeg_min: int = 30,
        jpeg_prob: float = 0.7,
        val: bool = False,
        resize_interp: str = "bilinear",
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotate_deg = rotate_deg
        self.perspective_distortion = perspective_distortion
        self.perspective_prob = perspective_prob
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.clean_prob = clean_prob
        self.resize_degrade_prob = resize_degrade_prob
        self.resize_degrade_min = resize_degrade_min
        self.resize_degrade_max = resize_degrade_max
        self.blur_max = blur_max
        self.blur_prob = blur_prob
        self.jpeg_min = jpeg_min
        self.jpeg_prob = jpeg_prob
        self.val = val
        # Resize interp for the "real" downscale paths (PIL): initial scaled
        # zoom, val-mode direct resize, and the post-crop fallback. LANCZOS
        # only valid on PIL — kept off the affine (rotate/perspective) and
        # off the corruption-aug (intentionally lossy) paths.
        self.resize_interp = resolve_resize_interp(resize_interp)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src_path, tgt_path = self.pairs[idx]
        src = Image.open(src_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        if self.val:
            sz = self.image_size
            src = TF.resize(src, [sz, sz], interpolation=self.resize_interp)
            tgt = TF.resize(tgt, [sz, sz], interpolation=self.resize_interp)
            return {"source": TF.to_tensor(src), "target": TF.to_tensor(tgt)}

        # Shared geometry (both src and tgt)
        src, tgt = self._shared_geom(src, tgt)

        # Source-only color jitter (PIL)
        src = self._color_jitter(src)

        src_t = TF.to_tensor(src)
        tgt_t = TF.to_tensor(tgt)

        # Source-only degradation (tensor)
        src_t = self._degrade(src_t)

        return {"source": src_t, "target": tgt_t}

    # ------------------------------------------------------------------

    def _shared_geom(self, src: Image.Image, tgt: Image.Image):
        sz = self.image_size

        # Random zoom: resize so shorter side = sz * scale
        scale = random.uniform(self.scale_min, self.scale_max)
        resize_to = max(sz + 4, int(round(sz * scale)))  # +4 to ensure room to crop
        src = TF.resize(src, resize_to, interpolation=self.resize_interp)
        tgt = TF.resize(tgt, resize_to, interpolation=self.resize_interp)

        # Random rotation — same angle for both
        if self.rotate_deg > 0:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            src = TF.rotate(src, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            tgt = TF.rotate(tgt, angle, interpolation=InterpolationMode.BILINEAR, fill=0)

        # Random perspective warp — same params for both
        if self.perspective_distortion > 0 and random.random() < self.perspective_prob:
            w, h = src.size
            sp, ep = _perspective_params(w, h, self.perspective_distortion)
            src = TF.perspective(src, sp, ep, interpolation=InterpolationMode.BILINEAR, fill=0)
            tgt = TF.perspective(tgt, sp, ep, interpolation=InterpolationMode.BILINEAR, fill=0)

        # Random crop to sz × sz
        w, h = src.size
        if h >= sz and w >= sz:
            top  = random.randint(0, h - sz)
            left = random.randint(0, w - sz)
            src = TF.crop(src, top, left, sz, sz)
            tgt = TF.crop(tgt, top, left, sz, sz)
        else:
            src = TF.resize(src, [sz, sz], interpolation=self.resize_interp)
            tgt = TF.resize(tgt, [sz, sz], interpolation=self.resize_interp)

        # Horizontal flip
        if random.random() < self.hflip_prob:
            src = TF.hflip(src)
            tgt = TF.hflip(tgt)

        return src, tgt

    def _color_jitter(self, img: Image.Image) -> Image.Image:
        """Brightness / contrast / saturation jitter — source only."""
        if self.brightness > 0:
            f = random.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(f)
        if self.contrast > 0:
            f = random.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(f)
        if self.saturation > 0:
            f = random.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)
            img = ImageEnhance.Color(img).enhance(f)
        return img

    def _degrade(self, img: torch.Tensor) -> torch.Tensor:
        """Source-only degradation, gated by clean_prob."""
        if random.random() < self.clean_prob:
            return img
        # Resize-down + resize-up (internet/pixelated style)
        if random.random() < self.resize_degrade_prob:
            h, w = img.shape[-2:]
            factor = random.uniform(self.resize_degrade_min, self.resize_degrade_max)
            sh = max(16, int(h * factor))
            sw = max(16, int(w * factor))
            img = TF.resize(img, [sh, sw], interpolation=InterpolationMode.BILINEAR, antialias=True)
            img = TF.resize(img, [h, w],   interpolation=InterpolationMode.BILINEAR, antialias=True)
        # Gaussian blur
        if self.blur_max > 0 and random.random() < self.blur_prob:
            sigma = random.uniform(0.5, self.blur_max)
            k = max(3, int(2 * math.ceil(3 * sigma) + 1) | 1)
            img = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)
        # JPEG
        if self.jpeg_min < 95 and random.random() < self.jpeg_prob:
            img = _jpeg_compress(img, random.randint(self.jpeg_min, 95))
        return img


def make_loader(pairs, image_size, batch_size, args, val=False):
    ds = ProgPairedDataset(
        pairs=pairs,
        image_size=image_size,
        scale_min=args.aug_scale_min,
        scale_max=args.aug_scale_max,
        rotate_deg=args.aug_rotate_deg,
        perspective_distortion=args.aug_perspective,
        perspective_prob=args.aug_perspective_prob,
        hflip_prob=0.5,
        brightness=args.aug_brightness,
        contrast=args.aug_contrast,
        saturation=args.aug_saturation,
        clean_prob=args.clean_prob,
        resize_degrade_prob=args.degrade_resize_prob,
        resize_degrade_min=args.degrade_resize_min,
        resize_degrade_max=args.degrade_resize_max,
        blur_max=args.corrupt_blur_max,
        blur_prob=args.corrupt_blur_prob,
        jpeg_min=args.corrupt_jpeg_min,
        jpeg_prob=args.corrupt_jpeg_prob,
        val=val,
        resize_interp=args.aug_resize_interp,
    )
    nw = args.num_workers
    dl_kw = dict(
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    return DataLoader(ds, shuffle=not val, **dl_kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr(step, total_steps, warmup_steps, lr_max, lr_min,
              num_cycles: int = 1, cycle_mult: float = 1.0):
    """Cosine schedule with optional SGDR-style warm restarts.

    num_cycles=1 (default): single smooth cosine from lr_max → lr_min over the
    post-warmup window — backward-compatible with the original schedule.

    num_cycles>1: split the post-warmup window into N cycles; each cycle is a
    full cosine descent from lr_max to lr_min before restarting. With
    cycle_mult > 1.0 cycles grow geometrically (e.g. 1k → 2k → 4k → ...).
    """
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    s = step - warmup_steps
    T_total = max(total_steps - warmup_steps, 1)

    if num_cycles <= 1:
        progress = s / T_total
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

    if cycle_mult == 1.0:
        T0 = T_total / num_cycles
    else:
        # T0 * (cycle_mult^N - 1) / (cycle_mult - 1) = T_total → T0 = T_total*(m-1)/(m^N-1)
        T0 = T_total * (cycle_mult - 1.0) / (cycle_mult ** num_cycles - 1.0)

    cycle_start, cycle_len = 0.0, T0
    for _ in range(num_cycles):
        if s < cycle_start + cycle_len:
            break
        cycle_start += cycle_len
        cycle_len *= cycle_mult
    progress = (s - cycle_start) / max(cycle_len, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def cosine_anneal(step: int, total_steps: int, start: float, end: float | None) -> float:
    """Cosine interpolation start → end over total_steps. None end = constant start."""
    if end is None or end == start:
        return start
    progress = min(step / max(total_steps, 1), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_anneal(step, total_steps, start, end):
    """Cosine interp start → end over total_steps. If end is None, returns start."""
    if end is None or end == start:
        return start
    progress = min(step / max(total_steps, 1), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="exp32: progressive 128→256→512 from-scratch training")
    p = apply_yaml_config(p)
    p.add_argument("data_root")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--model-ch", type=int, default=88)
    p.add_argument("--attn-resolutions", default="16,32,64")
    p.add_argument("--arch", default="unet", choices=["unet", "pixel_dit"],
                   help="Model architecture. 'unet' = Img2ImgDiffusionUNet (default); "
                        "'pixel_dit' = pure-pixel DiT (HiDream-O1 style, no UNet). "
                        "pixel_dit uses --dit-pixel-dim/--dit-pixel-layers/--dit-pixel-patch.")
    p.add_argument("--dit-pixel-dim", type=int, default=384,
                   help="Token width for pixel_dit (default 384 → ~49M params at 11 layers).")
    p.add_argument("--dit-pixel-layers", type=int, default=11,
                   help="Number of DiT blocks for pixel_dit (default 11 → ~49M total).")
    p.add_argument("--dit-pixel-patch", type=int, default=16,
                   help="Patch size for pixel_dit (default 16 → 256 tokens @ 256px, 1024 @ 512).")
    p.add_argument("--use-source-pyramid", action="store_true",
                   help="Enable in-model SourcePyramid + FiLM modulation of the decoder. "
                        "~2.4M extra params at mc=88. Zero-init FiLM → identity at init.")
    p.add_argument("--use-decoder-attn", action="store_true",
                   help="Mirror encoder attn on the decoder side: BottleneckAttention "
                        "at the same resolutions (attn_resolutions) operating on decoder "
                        "output channels. ~SD/SDXL convention.")
    p.add_argument("--use-dit-bottleneck", action="store_true",
                   help="Replace (mid_attn + mid2) with a stack of DiT blocks at the "
                        "bottleneck. mid1 still projects c4 → cm upstream. adaLN-zero "
                        "init → identity at step 0.")
    p.add_argument("--num-dit-blocks", type=int, default=4,
                   help="Number of DiT blocks in the bottleneck stack (default 4).")
    p.add_argument("--dit-mlp-ratio", type=float, default=4.0,
                   help="MLP hidden ratio inside each DiT block (default 4.0).")
    # phase schedule overrides (end steps)
    p.add_argument("--phase1-end", type=int, default=5_000,   help="Last step of 128px phase")
    p.add_argument("--phase2-end", type=int, default=25_000,  help="Last step of 256px phase")
    # batch sizes
    p.add_argument("--bs-128", type=int, default=64)
    p.add_argument("--bs-256", type=int, default=16)
    p.add_argument("--bs-512", type=int, default=4)
    # lr
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=500)
    p.add_argument("--lr-num-cycles", type=int, default=1,
                   help="Cosine cycles in the LR schedule. 1 = smooth single decay (default); "
                        ">1 = SGDR-style warm restarts.")
    p.add_argument("--lr-cycle-mult", type=float, default=1.0,
                   help="Geometric cycle-length growth factor (T_mult). 1.0 = equal cycles; "
                        ">1.0 = each cycle is `mult` times longer than the previous.")
    p.add_argument("--method", default="flow", choices=["flow", "diffusion"],
                   help="Training method. 'flow' = rectified flow matching (canonical "
                        "post-exp07); 'diffusion' = classic eps/v-prediction Gaussian "
                        "diffusion (legacy exp01-exp06 family).")
    p.add_argument("--diffusion-timesteps", type=int, default=1000,
                   help="(method=diffusion only) Number of timesteps in the noise "
                        "schedule. Standard 1000.")
    p.add_argument("--prediction-type", default="eps", choices=["eps", "v"],
                   help="(method=diffusion only) eps-prediction (legacy default) or "
                        "v-prediction (Salimans+Ho; better at high noise, more flow-like).")
    p.add_argument("--flow-sigma-noise", type=float, default=0.05,
                   help="(method=flow only) Off-path Gaussian noise sigma added to x_t "
                        "during training. Higher sigma = more stochasticity, less MSE-blur, "
                        "but inference must also start from source+sigma*noise to match.")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # lpips
    p.add_argument("--lpips-weight", type=float, default=0.2,
                   help="LPIPS weight at step 0. If --lpips-weight-end is set, "
                        "cosine-anneals to that value by args.steps.")
    p.add_argument("--lpips-weight-end", type=float, default=None,
                   help="Cosine-anneal LPIPS weight from --lpips-weight at step 0 to "
                        "this value at args.steps. Default None = constant weight.")
    p.add_argument("--lpips-weight-warmup-steps", type=int, default=0,
                   help="Linear LPIPS warmup: weight ramps from 0 to --lpips-weight over "
                        "these steps, then either holds or continues the cosine anneal. "
                        "Used to let pure flow-MSE establish spatial coherence before "
                        "perceptual force kicks in. 0 = no warmup (default).")
    p.add_argument("--lpips-aux-net", default="vgg", choices=["squeeze", "vgg", "alex"])
    p.add_argument("--contrastive-source-weight", type=float, default=0.0,
                   help="Margin contrastive: penalize predictions too close to source. "
                        "Loss += w * relu(margin - lpips(out, source)). 0 = off (default).")
    p.add_argument("--contrastive-source-margin", type=float, default=0.15,
                   help="Margin for the source-contrastive penalty. Below this lpips(out, source) "
                        "is penalized; above it the term is zero.")
    p.add_argument("--style-loss-weight", type=float, default=0.0,
                   help="VGG Gram-matrix style loss weight. 0 = off (default). "
                        "Rewards texture/style match independent of pixel alignment.")
    p.add_argument("--content-loss-weight", type=float, default=0.0,
                   help="VGG content (feature-L1) loss weight. 0 = off. Overlaps LPIPS-VGG; "
                        "usually leave 0 and rely on --lpips-weight + --style-loss-weight.")
    p.add_argument("--style-loss-layers", default="8,15,22",
                   help="VGG16 layer indices for the style/content feature loss.")
    p.add_argument("--source-dropout", type=float, default=0.0,
                   help="Probability of zeroing the source during training (CFG-style). "
                        "Enables CFG at inference. 0 = off (default).")
    # augmentation — shared geometry
    p.add_argument("--aug-resize-interp", default="bilinear",
                   choices=["bilinear", "bicubic", "lanczos", "nearest"],
                   help="PIL resize filter for the source-pool downscale and val resize. "
                        "LANCZOS preserves edge sharpness on >=2x downscales (e.g. 512->256). "
                        "Affine (rotate/perspective) and corruption-aug paths stay BILINEAR.")
    p.add_argument("--aug-scale-min", type=float, default=1.0,
                   help="Min random resize scale (1.0 = full image view)")
    p.add_argument("--aug-scale-max", type=float, default=2.5,
                   help="Max random resize scale (2.5 = tight zoom crop)")
    p.add_argument("--aug-rotate-deg", type=float, default=25.0,
                   help="Max rotation in degrees (±), applied to source+target")
    p.add_argument("--aug-perspective", type=float, default=0.15,
                   help="Perspective warp distortion strength [0–1]")
    p.add_argument("--aug-perspective-prob", type=float, default=0.5,
                   help="Probability of applying perspective warp per sample")
    # augmentation — source-only color jitter
    p.add_argument("--aug-brightness", type=float, default=0.3,
                   help="Brightness jitter ±amount on source")
    p.add_argument("--aug-contrast", type=float, default=0.3,
                   help="Contrast jitter ±amount on source")
    p.add_argument("--aug-saturation", type=float, default=0.3,
                   help="Saturation jitter ±amount on source")
    # source-only degradation (gated by clean_prob)
    p.add_argument("--clean-prob", type=float, default=0.2,
                   help="Probability of skipping all degradation (clean source)")
    p.add_argument("--degrade-resize-prob", type=float, default=0.3,
                   help="Probability of resize-down+up degradation")
    p.add_argument("--degrade-resize-min", type=float, default=0.25,
                   help="Min downscale factor for resize degradation")
    p.add_argument("--degrade-resize-max", type=float, default=0.75,
                   help="Max downscale factor for resize degradation")
    p.add_argument("--corrupt-blur-max", type=float, default=3.0)
    p.add_argument("--corrupt-jpeg-min", type=int, default=30)
    p.add_argument("--corrupt-blur-prob", type=float, default=0.7)
    p.add_argument("--corrupt-jpeg-prob", type=float, default=0.7)
    # val / logging
    p.add_argument("--amp", default="bf16", choices=["no", "bf16"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=5_000)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-image-size", type=int, default=0,
                   help="Validation/panel resolution. 0 (default) = use the final phase's "
                        "resolution (= the phase that contains args.steps; for single-phase "
                        "256 runs that's 256, for full progressive 128→256→512 it's 512). "
                        "Set explicitly to pin val at a fixed res.")
    p.add_argument("--panel-every", type=int, default=5_000)
    p.add_argument("--panel-keys", default="ffhq_002321,ffhq_002350,ffhq_002370,000942,000943,000921",
                   help="Comma-separated val sample stems to pin into every panel "
                        "snapshot. Default = three FFHQ portraits (from val_portraits, "
                        "3k-era runs) + three legacy close-up faces (from val, retained "
                        "for continuity). Stems missing from the dataset are silently "
                        "skipped. Empty string = first val batch (legacy behaviour).")
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--best-every", type=int, default=5_000,
                   help="Evaluate at this interval and save model_best.pt if val LPIPS improves")
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--exp-name", default="",
                   help="Short experiment tag prepended to every saved output filename "
                        "(e.g. 'exp33' → exp33_model.pt, exp33_panel_step_005000.png). "
                        "Default '' = no prefix.")
    p.add_argument("--resume", default=None,
                   help="Path to a previous checkpoint (*.pt). Loads model + EMA + step; "
                        "training continues from step+1 with the same CLI recipe. "
                        "Optimizer state isn't saved → Adam warmup re-acquires momentum "
                        "over the first ~100 steps.")
    p.add_argument("--outdir", default="out/exp32_prog512")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="nanoWarp")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Val helpers
# ---------------------------------------------------------------------------

def _sample_from_source(ema_model, diffusion, source, sample_steps, device, method="flow"):
    """Single rollout. Dispatches on method:
    - flow: Euler-ODE from x=source over t in [0,1]
    - diffusion: DDIM from x=randn over the model's full diffusion schedule
    """
    if method == "flow":
        ts = torch.linspace(0.0, 1.0, sample_steps + 1, device=device)
        x = source.clone()
        for j in range(sample_steps):
            t_cur = ts[j].expand(source.shape[0])
            v = ema_model(source, x, diffusion._scale_t(t_cur))
            x = x + float(ts[j + 1] - ts[j]) * v
        return x.clamp(0, 1)
    # diffusion: GaussianImageDiffusion.sample handles DDIM internally and
    # starts from N(0, I); source is passed as conditioning.
    H = source.shape[-1]
    ch = source.shape[1]
    out, _ = diffusion.sample(
        ema_model, source, image_size=H, channels=ch, sample_steps=sample_steps,
    )
    return out.clamp(0, 1)


@torch.no_grad()
def run_val(ema_model, diffusion, val_loader, args, device, step, outdir, wandb):
    """Run val twice per batch: once on clean source, once on a deterministically
    corrupted source (JPEG + small blur + resize). The corrupted-source LPIPS
    is a proxy 'robustness' metric — corruption-trained runs should keep the
    clean→corrupted gap small; clean-only runs (exp23/25) will show a big gap.
    """
    from src.img2img.metrics import ValidationMetrics, val_corrupt
    metrics_fn = ValidationMetrics(device)

    ema_model.eval()
    lpips_clean, ssim_clean = [], []
    lpips_corr,  ssim_corr  = [], []

    for i, batch in enumerate(val_loader):
        if i >= args.val_batches:
            break
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        # Clean source → output
        samples_clean = _sample_from_source(ema_model, diffusion, source, args.sample_steps, device, method=args.method)
        m_clean = metrics_fn.compute(samples_clean, target)
        lpips_clean.append(m_clean["lpips_squeeze"])
        ssim_clean.append(m_clean["ssim"])

        # Corrupted source → output (target is still the clean target — the
        # task being measured is "given a degraded source, can we still
        # reconstruct the clean target?")
        source_corr = val_corrupt(source)
        samples_corr = _sample_from_source(ema_model, diffusion, source_corr, args.sample_steps, device, method=args.method)
        m_corr = metrics_fn.compute(samples_corr, target)
        lpips_corr.append(m_corr["lpips_squeeze"])
        ssim_corr.append(m_corr["ssim"])

    mean_lpips_clean = sum(lpips_clean) / len(lpips_clean)
    mean_ssim_clean  = sum(ssim_clean)  / len(ssim_clean)
    mean_lpips_corr  = sum(lpips_corr)  / len(lpips_corr)
    mean_ssim_corr   = sum(ssim_corr)   / len(ssim_corr)
    delta_lpips = mean_lpips_corr - mean_lpips_clean
    print(f"[val] step={step}  clean lpips_sq={mean_lpips_clean:.4f} ssim={mean_ssim_clean:.4f}  |  "
          f"corrupted lpips_sq={mean_lpips_corr:.4f} ssim={mean_ssim_corr:.4f}  Δ={delta_lpips:+.4f}")

    fp = (args.exp_name + "_") if args.exp_name else ""
    with open(outdir / f"{fp}val_step{step:06d}.json", "w") as f:
        json.dump({
            "step": step,
            "lpips_sq": mean_lpips_clean,         # keep legacy keys for tooling
            "ssim": mean_ssim_clean,
            "lpips_sq_clean": mean_lpips_clean,
            "ssim_clean": mean_ssim_clean,
            "lpips_sq_corrupted": mean_lpips_corr,
            "ssim_corrupted": mean_ssim_corr,
            "lpips_sq_delta": delta_lpips,
        }, f, indent=2)

    if wandb is not None:
        wandb.log({
            "val/lpips_sq": mean_lpips_clean,
            "val/ssim": mean_ssim_clean,
            "val/lpips_sq_corrupted": mean_lpips_corr,
            "val/ssim_corrupted": mean_ssim_corr,
            "val/lpips_sq_delta": delta_lpips,
        }, step=step)

    return mean_lpips_clean


def build_panel_batch(pair_pools, panel_keys, image_size, device):
    """Pre-build a fixed batch from specific val stems (e.g. close-up faces) so
    every panel snapshot shows the same comparable samples across runs.

    `pair_pools` is a list of pair-list-objects (e.g. [val_pairs,
    val_portraits_pairs]); stems are searched across all pools in order.
    Stems matching multiple pools resolve to the first hit. Stems not
    found in any pool are silently skipped.

    Returns None when panel_keys is empty or no matching stems are found.
    """
    keys = [k.strip() for k in panel_keys.split(",") if k.strip()]
    if not keys:
        return None
    # Build stem → (source, target) map across all pools.
    stem_to_pair = {}
    for pool in pair_pools:
        for s, t in pool:
            if s.stem not in stem_to_pair:
                stem_to_pair[s.stem] = (s, t)
    # Preserve user-specified order of panel_keys.
    selected = [stem_to_pair[k] for k in keys if k in stem_to_pair]
    if not selected:
        return None
    ds = ProgPairedDataset(selected, image_size=image_size, val=True)
    sources = torch.stack([ds[i]["source"] for i in range(len(selected))]).to(device)
    targets = torch.stack([ds[i]["target"] for i in range(len(selected))]).to(device)
    found_stems = [s.stem for (s, _) in selected]
    return {"source": sources, "target": targets, "stems": found_stems}


@torch.no_grad()
def save_panel(ema_model, diffusion, val_loader, args, device, step, outdir):
    ema_model.eval()
    batch = next(iter(val_loader))
    source = batch["source"].to(device)[:4]
    target = batch["target"].to(device)[:4]
    samples = _sample_from_source(
        ema_model, diffusion, source, args.sample_steps, device, method=args.method,
    )
    fp = (args.exp_name + "_") if args.exp_name else ""
    save_val_panel(source.cpu(), target.cpu(), samples.cpu(), samples.cpu(),
                   outdir / f"{fp}panel_step_{step:06d}.png")


@torch.no_grad()
def save_face_panel(ema_model, diffusion, panel_batch, args, step, outdir):
    """Sample the pinned face-closeup batch and save to a separate filename."""
    if panel_batch is None:
        return
    ema_model.eval()
    source = panel_batch["source"]
    target = panel_batch["target"]
    samples = _sample_from_source(
        ema_model, diffusion, source, args.sample_steps, source.device, method=args.method,
    )
    fp = (args.exp_name + "_") if args.exp_name else ""
    save_val_panel(source.cpu(), target.cpu(), samples.cpu(), samples.cpu(),
                   outdir / f"{fp}face_panel_step_{step:06d}.png")


@torch.no_grad()
def infer_nat1(ema_model, diffusion, args, device, step, outdir):
    nat1_path = "/home/researcher/reference/nat1.mp4"
    try:
        import torchvision.io as tvio
        frames, _, _ = tvio.read_video(nat1_path, start_pts=0, end_pts=0.5, pts_unit="sec")
        if frames.shape[0] == 0:
            return
        frame_np = frames[0].numpy()
        frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        # Match the val/panel resolution so nat1 panels compare apples-to-apples
        # to the val curves.
        infer_size = args.val_image_size if args.val_image_size > 0 else (
            next(res for end, res, _ in PHASES if args.steps <= end)
        )
        frame_t = TF.resize(frame_t, [infer_size, infer_size], antialias=True)
        source = frame_t.unsqueeze(0).to(device)

        ema_model.eval()
        result = _sample_from_source(
            ema_model, diffusion, source, args.sample_steps, device, method=args.method,
        )

        grid = torch.cat([source.cpu(), result.cpu()], dim=3)
        fp = (args.exp_name + "_") if args.exp_name else ""
        TF.to_pil_image(grid[0]).save(outdir / f"{fp}nat1_step_{step:06d}.png")
        print(f"[nat1] saved {fp}nat1_step_{step:06d}.png")
    except Exception as e:
        print(f"[warn] nat1 inference failed: {e}")


def save_checkpoint(model, ema, method_cfg, args, step, path):
    # The UNet constructor takes architecture flags that this script hardcodes
    # (rather than exposing via CLI). Save them explicitly so validate.py and
    # downstream loaders don't have to infer them from state_dict shapes.
    config = dict(vars(args))
    config.setdefault("source_in_stem", True)
    config.setdefault("no_source_encoder", True)   # use_source_encoder = not no_source_encoder
    config.setdefault("upsample_type", "resize_conv")
    config.setdefault("image_size", 512)            # construction-time size (model handles any input)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "config": config,
        "method": args.method,
    }
    if args.method == "flow":
        payload["flow"] = method_cfg.__dict__
    else:
        payload["diffusion"] = method_cfg.__dict__
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- wandb ---
    wandb = None
    if args.wandb:
        import os
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        import wandb as _wandb
        if api_key:
            _wandb.login(key=api_key, relogin=True)
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] or None
        run_name = args.wandb_run_name or outdir.name
        try:
            _wandb.init(project=args.wandb_project, name=run_name, tags=tags,
                        config=vars(args), dir=str(outdir))
            wandb = _wandb
            print(f"wandb run: {wandb.run.name}  ({wandb.run.url})")
            # Save run id + project so validate.py can resume into the same
            # run and append final-val metrics — useful if Colab dies after
            # training but before the final-val finishes writing stdout.
            (outdir / "wandb_run.txt").write_text(
                f"{wandb.run.project}\n{wandb.run.id}\n", encoding="utf-8"
            )
        except Exception as e:
            print(f"[warn] wandb init failed: {type(e).__name__}: {e} — continuing without wandb")

    # --- build phases ---
    global PHASES
    PHASES = [
        (args.phase1_end,  128, args.bs_128),
        (args.phase2_end,  256, args.bs_256),
        (args.steps,       512, args.bs_512),
    ]

    # --- model ---
    attn_res = tuple(int(x) for x in args.attn_resolutions.split(",") if x.strip())
    if args.arch == "pixel_dit":
        model = PixelDiT(
            in_channels=3,
            out_channels=3,
            patch_size=args.dit_pixel_patch,
            dim=args.dit_pixel_dim,
            num_layers=args.dit_pixel_layers,
            num_heads=max(1, args.dit_pixel_dim // 64),  # head_dim ≈ 64 for flash-attn
            mlp_ratio=4.0,
            source_in_stem=True,
        ).to(device)
    else:
        model = Img2ImgDiffusionUNet(
            model_ch=args.model_ch,
            pretrained_source_encoder=False,
            source_in_stem=True,
            use_source_encoder=False,
            upsample_type="resize_conv",
            attn_resolutions=attn_res,
            image_size=512,   # model built at 512; handles smaller inputs fine
            use_source_pyramid=args.use_source_pyramid,
            use_decoder_attn=args.use_decoder_attn,
            use_dit_bottleneck=args.use_dit_bottleneck,
            num_dit_blocks=args.num_dit_blocks,
            dit_mlp_ratio=args.dit_mlp_ratio,
        ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arch={args.arch}  params total={total:,}  trainable={trainable:,}")

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )

    # Build the training method object. `diffusion` is the legacy var name
    # (predates the flow rewrite); it now holds either a flow or diffusion
    # instance, kept identical-shaped 8-tuple training_loss return so the
    # downstream loop doesn't need to branch on method.
    if args.method == "flow":
        method_cfg = FlowConfig(timesteps=1000, sigma_noise=args.flow_sigma_noise, method="flow")
        diffusion = RectifiedImageFlow(method_cfg, device)
    else:
        method_cfg = DiffusionConfig(
            timesteps=args.diffusion_timesteps,
            prediction_type=args.prediction_type,
        )
        diffusion = GaussianImageDiffusion(method_cfg, device)
    print(f"method={args.method}  method_cfg={method_cfg.__dict__}")

    aux_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=args.lpips_aux_net, normalize=True
    ).to(device)
    for p in aux_lpips.parameters():
        p.requires_grad_(False)
    print(f"lpips_aux_net={args.lpips_aux_net} weight={args.lpips_weight}")

    # Optional VGG style / content loss (Gram-matrix style transfer, Gatys/Johnson).
    # Used to push outputs toward target *texture statistics* independent of
    # pixel alignment — complements LPIPS which is pixel-aligned.
    style_loss_fn = None
    if args.style_loss_weight > 0 or args.content_loss_weight > 0:
        from src.img2img.feature_loss import VGGFeatureLoss
        layers = tuple(int(x) for x in args.style_loss_layers.split(",") if x.strip())
        style_loss_fn = VGGFeatureLoss(
            layers=layers,
            content_weight=args.content_loss_weight,
            style_weight=args.style_loss_weight,
        ).to(device)
        print(f"vgg_feature_loss  layers={layers}  content={args.content_loss_weight}  style={args.style_loss_weight}")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    use_amp = amp_dtype is not None and device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    print(f"amp={args.amp}  autocast={use_amp}")

    # --- data ---
    data_root = Path(args.data_root)
    train_pairs = _list_pairs(data_root, "train")
    val_pairs   = _list_pairs(data_root, "val")
    # 3k-era dataset has an extra val_portraits split with FFHQ frontal faces.
    # Used only as an additional source for the panel batch (so we can pin
    # frontal portrait stems); the in-loop val loop still uses `val/`.
    try:
        val_portraits_pairs = _list_pairs(data_root, "val_portraits")
        print(f"dataset: {len(train_pairs)} train, {len(val_pairs)} val, "
              f"{len(val_portraits_pairs)} val_portraits pairs")
    except (ValueError, FileNotFoundError):
        val_portraits_pairs = []
        print(f"dataset: {len(train_pairs)} train, {len(val_pairs)} val pairs")
    print(f"aug  scale=[{args.aug_scale_min},{args.aug_scale_max}]  "
          f"rotate=±{args.aug_rotate_deg}°  perspective={args.aug_perspective}@p={args.aug_perspective_prob}")
    print(f"aug  brightness=±{args.aug_brightness}  contrast=±{args.aug_contrast}  saturation=±{args.aug_saturation}")
    print(f"degrade  clean_prob={args.clean_prob}  resize_prob={args.degrade_resize_prob}@[{args.degrade_resize_min},{args.degrade_resize_max}]  "
          f"blur_max={args.corrupt_blur_max}@p={args.corrupt_blur_prob}  jpeg_min={args.corrupt_jpeg_min}@p={args.corrupt_jpeg_prob}")

    # Val/panel loader at the final-phase resolution (or --val-image-size if set).
    # Keeping val res fixed across the whole run makes val curves comparable
    # step-to-step. For the original progressive 128→256→512 recipe that's
    # 512; for the single-phase 256 runs (exp33+) it's 256.
    final_image_size = next(res for end, res, _ in PHASES if args.steps <= end)
    val_image_size = args.val_image_size if args.val_image_size > 0 else final_image_size
    val_bs = next(bs for end, _, bs in PHASES if args.steps <= end)
    val_loader = make_loader(val_pairs, val_image_size, val_bs, args, val=True)

    # Pinned face-closeup panel batch (always the same val samples across
    # steps and runs → directly comparable face crops).
    face_panel_batch = build_panel_batch(
        [val_pairs, val_portraits_pairs], args.panel_keys, val_image_size, device,
    )
    if face_panel_batch is not None:
        print(f"[face_panel] pinned val stems: {face_panel_batch['stems']}")
    else:
        print(f"[face_panel] none — panel_keys='{args.panel_keys}' matched no val stems")
    print(f"[val] loader built at image_size={val_image_size}  bs={val_bs}  "
          f"(final-phase res; override with --val-image-size)")

    # --- training state ---
    losses: list[float] = []
    best_lpips = float("inf")
    cur_phase_idx = -1
    train_iter = None
    start_step = 1

    # Optional resume: load model + EMA + step from a previous checkpoint.
    # The cosine LR schedule reads `step` so the LR resumes at the right
    # fraction of the run; optimizer state is intentionally fresh.
    if args.resume:
        rckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(rckpt["model"])
        if "ema_model" in rckpt:
            ema.model.load_state_dict(rckpt["ema_model"])
        start_step = int(rckpt.get("step", 0)) + 1
        print(f"[resume] loaded {args.resume} (step={start_step - 1}) → continuing from step {start_step}")

    print(f"phases: {PHASES}")
    print(f"training {start_step - 1}→{args.steps} steps")

    t_start = time.monotonic()
    t_window = t_start
    step_window = 0

    for step in range(start_step, args.steps + 1):
        # --- phase transition ---
        new_phase_idx = next(i for i, (end, _, _) in enumerate(PHASES) if step <= end)
        if new_phase_idx != cur_phase_idx:
            cur_phase_idx = new_phase_idx
            _, img_size, bs = PHASES[cur_phase_idx]
            train_loader = make_loader(train_pairs, img_size, bs, args, val=False)
            train_iter = cycle(train_loader)
            print(f"[phase {cur_phase_idx+1}] step={step}  image_size={img_size}  bs={bs}")
            if wandb is not None:
                wandb.log({"phase": cur_phase_idx + 1, "train_image_size": img_size}, step=step)

        model.train()
        lr = cosine_lr(step, args.steps, args.lr_warmup_steps, args.lr, args.lr_min,
                       num_cycles=args.lr_num_cycles, cycle_mult=args.lr_cycle_mult)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = next(train_iter)
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        # Per-step LPIPS weight (cosine anneal if --lpips-weight-end set).
        if args.lpips_weight_warmup_steps > 0 and step < args.lpips_weight_warmup_steps:
            # Linear ramp from 0 to args.lpips_weight over warmup_steps.
            lpips_w = args.lpips_weight * (step / max(args.lpips_weight_warmup_steps, 1))
        else:
            lpips_w = cosine_anneal(step, args.steps, args.lpips_weight, args.lpips_weight_end)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss_kwargs = dict(
                aux_lpips=aux_lpips, aux_lpips_weight=lpips_w,
                source_dropout=args.source_dropout,
            )
            if args.method == "flow":
                loss_kwargs["contrastive_source_weight"] = args.contrastive_source_weight
                loss_kwargs["contrastive_source_margin"] = args.contrastive_source_margin
            loss, t, x_t, _noise, _model_out, x0_hat, flow_loss, lpips_loss = diffusion.training_loss(
                model, source, target, **loss_kwargs,
            )
            # Optional VGG feature/style loss on the model's target prediction.
            if style_loss_fn is not None:
                vgg_terms = style_loss_fn(x0_hat.clamp(0, 1), target)
                loss = loss + vgg_terms["total"]

        loss_val = float(loss.item())
        # Track every step (incl. skipped) so the rolling mean self-corrects
        # after a regime shift. Old code appended only after the skip checks,
        # which froze recent_mean and caused a dead-loop when the loss entered
        # a new low-loss regime (e.g. diffusion-eps without LPIPS smoothing:
        # 0.0001 -> 0.0012 is 12x but harmless, yet the frozen mean kept
        # tripping the threshold forever).
        losses.append(loss_val)
        if not math.isfinite(loss_val):
            print(f"step {step:6d} | SKIP non-finite loss={loss_val}")
            optimizer.zero_grad(set_to_none=True)
            continue
        if len(losses) >= 11:
            baseline = losses[-51:-1]  # exclude current
            recent_mean = sum(baseline) / len(baseline)
            # Spike check: large RELATIVE jump AND non-trivial ABSOLUTE size.
            # The absolute floor (0.05) keeps tiny-loss regimes from
            # generating spurious skips.
            if recent_mean > 0 and loss_val > 10.0 * recent_mean and loss_val > 0.05:
                print(f"step {step:6d} | SKIP spike {loss_val/recent_mean:.1f}x abs={loss_val:.4f}")
                optimizer.zero_grad(set_to_none=True)
                continue

        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip_norm
            )
        optimizer.step()
        ema.update(model)

        step_window += 1
        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
            _, img_size, _ = PHASES[cur_phase_idx]
            # Rate from this window only — survives phase transitions where
            # step time changes with resolution. ETA recomputes each window.
            now = time.monotonic()
            window_dt = max(now - t_window, 1e-6)
            sec_per_step = window_dt / max(step_window, 1)
            remaining = max(args.steps - step, 0)
            eta_sec = remaining * sec_per_step
            elapsed_sec = now - t_start

            def _fmt(s):
                s = int(s)
                h, s = divmod(s, 3600)
                m, s = divmod(s, 60)
                return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

            t_window = now
            step_window = 0

            print(f"step={step:6d}/{args.steps}  loss={avg:.5f}  flow={float(flow_loss):.5f}"
                  f"  lpips={float(lpips_loss):.5f}  lr={lr:.2e}  res={img_size}"
                  f"  {1.0/sec_per_step:.2f}it/s  elapsed={_fmt(elapsed_sec)}  eta={_fmt(eta_sec)}")
            if wandb is not None:
                wandb.log({"loss": avg, "flow_loss": float(flow_loss),
                           "lpips_loss": float(lpips_loss), "lr": lr,
                           "lpips_weight": lpips_w,
                           "throughput/it_per_s": 1.0 / sec_per_step,
                           "throughput/eta_sec": eta_sec}, step=step)

        fp = (args.exp_name + "_") if args.exp_name else ""
        if step % args.checkpoint_every == 0:
            ckpt_path = outdir / f"{fp}model_step_{step:06d}.pt"
            save_checkpoint(model, ema, method_cfg, args, step, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

        if step % args.val_every == 0 or step % args.best_every == 0:
            val_lpips = run_val(ema.model, diffusion, val_loader, args, device, step, outdir, wandb)
            infer_nat1(ema.model, diffusion, args, device, step, outdir)

            if val_lpips < best_lpips:
                best_lpips = val_lpips
                best_path = outdir / f"{fp}model_best.pt"
                save_checkpoint(model, ema, method_cfg, args, step, best_path)
                print(f"[best] new best lpips_sq={best_lpips:.4f}  saved {best_path}")
                if wandb is not None:
                    wandb.log({"val/best_lpips_sq": best_lpips}, step=step)

        if step % args.panel_every == 0:
            save_panel(ema.model, diffusion, val_loader, args, device, step, outdir)
            save_face_panel(ema.model, diffusion, face_panel_batch, args, step, outdir)

    # Final save
    fp = (args.exp_name + "_") if args.exp_name else ""
    final_path = outdir / f"{fp}model.pt"
    save_checkpoint(model, ema, method_cfg, args, args.steps, final_path)
    print(f"[done] saved {final_path}  best_lpips_sq={best_lpips:.4f}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
