"""Temporal dataset v2 for WAN-style first-frame conditioned video-consistent img2img.

Synthesizes T-frame clips from still paired images via smooth pan/zoom camera
trajectories. Source and target receive the same trajectory, preserving
pixel-level correspondence.

WAN-style anchor frame conditioning:
    With probability anchor_prob, the first frame of a clip is designated as
    an "anchor" frame (anchor_mask[0] = True). The training script then feeds
    the clean target frame as noisy_target for that position, and passes mask=0
    via model.mask_proj. This teaches the model to use the anchor as a
    temporal reference during inference — enabling long-video generation by
    reinserting the last frame of each chunk as the first frame of the next.

Each item yields:
    source      : (T, 3, H, W)  float32  [0, 1]
    target      : (T, 3, H, W)  float32  [0, 1]
    anchor_mask : (T,)           bool     True = anchor (given clean frame)
    key         : str
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class TemporalAugConfig:
    image_size: int = 256
    # resize full image to image_size * spatial_scale before sampling crops
    spatial_scale: float = 2.0
    num_frames: int = 4           # T — frames per clip
    max_pan_frac: float = 0.25    # max total pan as fraction of (canvas - crop)
    zoom_range: tuple[float, float] = field(default_factory=lambda: (0.90, 1.10))
    horizontal_flip_prob: float = 0.5
    # WAN-style: probability that frame 0 is designated as an anchor frame
    anchor_prob: float = 0.5


def _list_pairs(
    root: Path,
    source_dir: str = "source",
    target_dir: str = "target",
    split: str | None = "train",
) -> list[tuple[str, Path, Path]]:
    base = root / split if split and (root / split).exists() else root
    src_root = base / source_dir
    tgt_root = base / target_dir

    src_files = {p.stem: p for p in sorted(src_root.iterdir())
                 if p.is_file() and p.suffix.lower() in IMG_EXTS}
    tgt_files = {p.stem: p for p in sorted(tgt_root.iterdir())
                 if p.is_file() and p.suffix.lower() in IMG_EXTS}
    keys = sorted(set(src_files) & set(tgt_files))
    if not keys:
        raise ValueError(f"No paired files in {src_root} and {tgt_root}")
    return [(k, src_files[k], tgt_files[k]) for k in keys]


def _affine_crop(
    img: Image.Image,
    y0: float,
    x0: float,
    crop_h: int,
    crop_w: int,
    zoom: float,
) -> Image.Image:
    """Crop a region starting at (y0, x0) with optional zoom, resize to (crop_h, crop_w)."""
    region_h = int(round(crop_h / zoom))
    region_w = int(round(crop_w / zoom))
    y0i = max(0, min(int(round(y0)), img.height - region_h))
    x0i = max(0, min(int(round(x0)), img.width - region_w))
    cropped = TF.crop(img, y0i, x0i, region_h, region_w)
    if region_h != crop_h or region_w != crop_w:
        cropped = TF.resize(cropped, [crop_h, crop_w], interpolation=InterpolationMode.BILINEAR)
    return cropped


class TemporalPairedDataset(Dataset):
    """Yields T-frame clips synthesized from still paired images.

    Returns a dict with:
        source      : (T, 3, H, W)  float32  [0, 1]
        target      : (T, 3, H, W)  float32  [0, 1]
        anchor_mask : (T,)           bool     True = anchor (clean given frame)
        key         : str
    """

    def __init__(
        self,
        root: str | Path,
        source_dir: str = "source",
        target_dir: str = "target",
        split: str | None = "train",
        config: TemporalAugConfig | None = None,
    ):
        self.root = Path(root)
        self.cfg = config or TemporalAugConfig()
        self.items = _list_pairs(self.root, source_dir, target_dir, split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        key, src_path, tgt_path = self.items[idx]
        cfg = self.cfg
        T = cfg.num_frames
        img_size = cfg.image_size

        rng = random.Random(idx)

        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        canvas_size = int(round(img_size * cfg.spatial_scale))
        src_img = TF.resize(src_img, [canvas_size, canvas_size], interpolation=InterpolationMode.BILINEAR)
        tgt_img = TF.resize(tgt_img, [canvas_size, canvas_size], interpolation=InterpolationMode.BILINEAR)

        if rng.random() < cfg.horizontal_flip_prob:
            src_img = TF.hflip(src_img)
            tgt_img = TF.hflip(tgt_img)

        max_offset = canvas_size - img_size
        max_pan = max_offset * cfg.max_pan_frac

        y_start = rng.uniform(0, max_offset)
        x_start = rng.uniform(0, max_offset)
        y_end = min(max(0.0, y_start + rng.uniform(-max_pan, max_pan)), max_offset)
        x_end = min(max(0.0, x_start + rng.uniform(-max_pan, max_pan)), max_offset)
        zoom_start = rng.uniform(*cfg.zoom_range)
        zoom_end = rng.uniform(*cfg.zoom_range)

        frames_src = []
        frames_tgt = []

        for i in range(T):
            alpha = i / (T - 1) if T > 1 else 0.0
            y = y_start + alpha * (y_end - y_start)
            x = x_start + alpha * (x_end - x_start)
            z = zoom_start + alpha * (zoom_end - zoom_start)
            frames_src.append(TF.to_tensor(_affine_crop(src_img, y, x, img_size, img_size, z)))
            frames_tgt.append(TF.to_tensor(_affine_crop(tgt_img, y, x, img_size, img_size, z)))

        source_t = torch.stack(frames_src)  # (T, 3, H, W)
        target_t = torch.stack(frames_tgt)

        # WAN-style: randomly anchor the first frame
        anchor_mask = torch.zeros(T, dtype=torch.bool)
        if rng.random() < cfg.anchor_prob:
            anchor_mask[0] = True

        return {"source": source_t, "target": target_t, "anchor_mask": anchor_mask, "key": key}
