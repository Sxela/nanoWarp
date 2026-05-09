from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class AugmentConfig:
    image_size: int = 128
    resize_min_scale: float = 1.10
    horizontal_flip_prob: float = 0.5
    rotation_deg: float = 5.0
    scale_jitter: float = 0.10
    translate_ratio: float = 0.05
    source_brightness: float = 0.20
    source_contrast: float = 0.20
    source_saturation: float = 0.15
    source_hue: float = 0.03
    source_gamma: float = 0.15
    source_blur_prob: float = 0.15
    source_blur_radius: float = 1.0


def _list_images(folder: Path) -> dict[str, Path]:
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files[p.stem] = p
    return files


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _rand_uniform(rng: random.Random, amount: float) -> float:
    return rng.uniform(max(0.0, 1.0 - amount), 1.0 + amount)


class PairedImageAugment:
    def __init__(self, config: AugmentConfig | None = None):
        self.config = config or AugmentConfig()

    def _shared_geom(self, src: Image.Image, tgt: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
        cfg = self.config
        target = cfg.image_size
        resize_to = int(round(target * cfg.resize_min_scale))

        src = TF.resize(src, resize_to, interpolation=InterpolationMode.BILINEAR)
        tgt = TF.resize(tgt, resize_to, interpolation=InterpolationMode.BILINEAR)

        scale = rng.uniform(1.0 - cfg.scale_jitter, 1.0 + cfg.scale_jitter)
        angle = rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)
        max_translate = int(round(target * cfg.translate_ratio))
        translate = (rng.randint(-max_translate, max_translate), rng.randint(-max_translate, max_translate))

        width, height = src.size
        center = (width * 0.5, height * 0.5)
        src = TF.affine(src, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, center=center)
        tgt = TF.affine(tgt, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, center=center)

        if rng.random() < cfg.horizontal_flip_prob:
            src = TF.hflip(src)
            tgt = TF.hflip(tgt)

        if src.height > target and src.width > target:
            i = rng.randint(0, src.height - target)
            j = rng.randint(0, src.width - target)
            h = w = target
        else:
            src = TF.resize(src, [target, target], interpolation=InterpolationMode.BILINEAR)
            tgt = TF.resize(tgt, [target, target], interpolation=InterpolationMode.BILINEAR)
            return src, tgt

        src = TF.crop(src, i, j, h, w)
        tgt = TF.crop(tgt, i, j, h, w)
        return src, tgt

    def _source_only_style(self, src: Image.Image, rng: random.Random) -> Image.Image:
        cfg = self.config
        src = ImageEnhance.Brightness(src).enhance(_rand_uniform(rng, cfg.source_brightness))
        src = ImageEnhance.Contrast(src).enhance(_rand_uniform(rng, cfg.source_contrast))
        src = ImageEnhance.Color(src).enhance(_rand_uniform(rng, cfg.source_saturation))

        hue_shift = rng.uniform(-cfg.source_hue, cfg.source_hue)
        if abs(hue_shift) > 1e-6:
            hsv = src.convert("HSV")
            h, s, v = hsv.split()
            offset = int(hue_shift * 255)
            h = h.point(lambda px: (px + offset) % 256)
            src = Image.merge("HSV", (h, s, v)).convert("RGB")

        gamma = rng.uniform(max(0.1, 1.0 - cfg.source_gamma), 1.0 + cfg.source_gamma)
        src = TF.adjust_gamma(src, gamma=gamma)

        if rng.random() < cfg.source_blur_prob:
            radius = rng.uniform(0.1, cfg.source_blur_radius)
            src = src.filter(ImageFilter.GaussianBlur(radius=radius))
        return src

    def __call__(self, src: Image.Image, tgt: Image.Image, seed: int | None = None) -> tuple[Image.Image, Image.Image, Image.Image]:
        rng = random.Random(seed)
        src_geom, tgt_geom = self._shared_geom(src, tgt, rng)
        src_style = self._source_only_style(src_geom.copy(), rng)
        return src_style, tgt_geom, src_geom


class PairedImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        source_dir: str = "source",
        target_dir: str = "target",
        augment: PairedImageAugment | None = None,
        split: str | None = None,
    ):
        self.root = Path(root)
        base_root = self.root / split if split and (self.root / split).exists() else self.root
        self.source_root = base_root / source_dir
        self.target_root = base_root / target_dir
        self.augment = augment or PairedImageAugment()

        src_files = _list_images(self.source_root)
        tgt_files = _list_images(self.target_root)
        keys = sorted(set(src_files) & set(tgt_files))
        if not keys:
            raise ValueError(f"No paired files found in {self.source_root} and {self.target_root}")
        self.items = [(k, src_files[k], tgt_files[k]) for k in keys]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        key, src_path, tgt_path = self.items[idx]
        src = _load_rgb(src_path)
        tgt = _load_rgb(tgt_path)

        src_aug, tgt_aug, src_geom = self.augment(src, tgt)
        return {
            "key": key,
            "source": TF.to_tensor(src_aug),
            "target": TF.to_tensor(tgt_aug),
            "source_geom": TF.to_tensor(src_geom),
        }


class IdentityPairedAugment:
    def __init__(self, image_size: int = 128):
        self.image_size = image_size

    def __call__(self, src: Image.Image, tgt: Image.Image, seed: int | None = None):
        src = TF.resize(src, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        tgt = TF.resize(tgt, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        return src, tgt, src.copy()


def build_train_val_datasets(
    train_root: str | Path,
    val_root: str | Path | None = None,
    image_size: int = 128,
    train_split: str | None = None,
    val_split: str | None = None,
):
    train_ds = PairedImageDataset(train_root, augment=PairedImageAugment(AugmentConfig(image_size=image_size)), split=train_split)
    resolved_val_root = val_root or train_root
    val_ds = PairedImageDataset(resolved_val_root, augment=IdentityPairedAugment(image_size=image_size), split=val_split)
    return train_ds, val_ds
