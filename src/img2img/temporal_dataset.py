"""Temporal dataset for video-consistent img2img finetuning.

Synthesizes short video clips from still paired images by applying smooth
pan/zoom camera trajectories. Source and target receive the same trajectory,
preserving pixel-level correspondence. Optical flow is computed analytically
from the affine parameters — no RAFT needed.

Each item yields a 2T-frame clip (two consecutive chunks of T frames), flow
vectors between adjacent frames, and the chunk-boundary flow. The training
loop splits the clip into chunk_a (frames 0..T-1) and chunk_b (frames T..2T-1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class TemporalAugConfig:
    image_size: int = 256
    # resize the full image to image_size * spatial_scale before sampling crops
    spatial_scale: float = 2.0
    num_frames: int = 8          # T — frames per chunk; dataset yields 2T frames
    max_pan_frac: float = 0.25   # max total pan as fraction of (resized - crop) range
    zoom_range: tuple[float, float] = field(default_factory=lambda: (0.90, 1.10))
    horizontal_flip_prob: float = 0.5
    # If True, augmentation is seeded per-index → identical clip every epoch
    # (use for validation). If False, uses the worker's RNG → fresh trajectory
    # each call. Default False so training gets real augmentation diversity.
    deterministic: bool = False


def _list_pairs(root: Path, source_dir: str = "source", target_dir: str = "target",
                split: str | None = "train") -> list[tuple[str, Path, Path]]:
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


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _affine_crop(img: Image.Image, y0: float, x0: float, crop_h: int, crop_w: int,
                 zoom: float) -> Image.Image:
    """Crop a region starting at (y0, x0) with optional zoom, resize to (crop_h, crop_w)."""
    # compute actual region size before resizing
    region_h = int(round(crop_h / zoom))
    region_w = int(round(crop_w / zoom))
    y0i = int(round(y0))
    x0i = int(round(x0))
    # clamp so we never exceed image bounds
    y0i = max(0, min(y0i, img.height - region_h))
    x0i = max(0, min(x0i, img.width - region_w))
    cropped = TF.crop(img, y0i, x0i, region_h, region_w)
    if region_h != crop_h or region_w != crop_w:
        cropped = TF.resize(cropped, [crop_h, crop_w], interpolation=InterpolationMode.BILINEAR)
    return cropped


class TemporalPairedDataset(Dataset):
    """Yields 2T-frame clips synthesized from still paired images.

    Returns a dict with:
        source  : (2T, 3, H, W) float32 in [0, 1]
        target  : (2T, 3, H, W) float32 in [0, 1]
        flow    : (2T-1, 2) float32 — per-transition (dx, dy) in output pixels
                  (positive dx = content moves right in next frame)
        key     : str
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
        total_frames = T * 2
        img_size = cfg.image_size

        # Train: module-level random → worker-seeded, varies every epoch.
        # Val: Random(idx) → identical clip per index, comparable curves.
        rng = random.Random(idx) if cfg.deterministic else random

        src_img = _load_rgb(src_path)
        tgt_img = _load_rgb(tgt_path)

        # resize to spatial_scale * image_size (canvas for the camera to pan over)
        canvas_size = int(round(img_size * cfg.spatial_scale))
        src_img = TF.resize(src_img, [canvas_size, canvas_size], interpolation=InterpolationMode.BILINEAR)
        tgt_img = TF.resize(tgt_img, [canvas_size, canvas_size], interpolation=InterpolationMode.BILINEAR)

        # horizontal flip (same for source and target)
        if rng.random() < cfg.horizontal_flip_prob:
            src_img = TF.hflip(src_img)
            tgt_img = TF.hflip(tgt_img)

        # sample start/end crop top-left and zoom
        # valid range for top-left: [0, canvas_size - img_size]
        max_offset = canvas_size - img_size  # e.g. 512 - 256 = 256
        max_pan = max_offset * cfg.max_pan_frac  # max total displacement

        y_start = rng.uniform(0, max_offset)
        x_start = rng.uniform(0, max_offset)
        # end position: start + random pan, clamped to valid range
        y_end = min(max(0.0, y_start + rng.uniform(-max_pan, max_pan)), max_offset)
        x_end = min(max(0.0, x_start + rng.uniform(-max_pan, max_pan)), max_offset)

        zoom_start = rng.uniform(*cfg.zoom_range)
        zoom_end = rng.uniform(*cfg.zoom_range)

        # build per-frame (y, x, zoom) by linear interpolation
        frames_src = []
        frames_tgt = []
        positions: list[tuple[float, float, float]] = []  # (y, x, zoom) per frame

        for i in range(total_frames):
            alpha = i / (total_frames - 1) if total_frames > 1 else 0.0
            y = y_start + alpha * (y_end - y_start)
            x = x_start + alpha * (x_end - x_start)
            z = zoom_start + alpha * (zoom_end - zoom_start)
            positions.append((y, x, z))

            src_crop = _affine_crop(src_img, y, x, img_size, img_size, z)
            tgt_crop = _affine_crop(tgt_img, y, x, img_size, img_size, z)
            frames_src.append(TF.to_tensor(src_crop))
            frames_tgt.append(TF.to_tensor(tgt_crop))

        source_t = torch.stack(frames_src)  # (2T, 3, H, W)
        target_t = torch.stack(frames_tgt)

        # analytic flow: for pure translation, content displacement = -(crop shift)
        # flow[i] = (dx, dy) such that content at pixel p in frame i appears at p + flow[i] in frame i+1
        # crop shift right (+x) → content moves left in image → flow_x = -(x_{i+1} - x_i) * zoom factor
        # We return flow in output-pixel units (after zoom resize).
        flow_list = []
        for i in range(total_frames - 1):
            y_i, x_i, z_i = positions[i]
            y_n, x_n, z_n = positions[i + 1]
            # crop origin shifts: positive = window moved right/down → content moves left/up
            # average zoom for this transition
            z_avg = (z_i + z_n) * 0.5
            # content displacement in output pixels = -(crop_shift / zoom * zoom) = -crop_shift
            # (zoom affects how much of the canvas each pixel covers, but after resize to img_size
            #  the crop shift in canvas pixels maps to the same shift in output pixels when zoom~1)
            flow_x = -(x_n - x_i)
            flow_y = -(y_n - y_i)
            flow_list.append(torch.tensor([flow_x, flow_y], dtype=torch.float32))

        flow = torch.stack(flow_list)  # (2T-1, 2)

        return {"source": source_t, "target": target_t, "flow": flow, "key": key}


def warp_by_translation(img: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Warp img by a per-sample constant (dx, dy) translation in pixels.

    img : (B, C, H, W)
    dx  : (B,) — positive = content shifts right
    dy  : (B,) — positive = content shifts down
    Returns warped image of same shape.
    """
    B, C, H, W = img.shape
    # normalized displacement: pixel / (size/2)
    tx = dx / (W / 2.0)
    ty = dy / (H / 2.0)
    # theta for affine_grid: [[1, 0, tx], [0, 1, ty]]
    theta = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
