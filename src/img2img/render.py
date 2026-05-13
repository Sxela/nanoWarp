from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    arr = x.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(arr)


def _with_label(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 24), "white")
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), text, fill=(20, 20, 20))
    return canvas


def save_training_panel(source: torch.Tensor, target: torch.Tensor, noisy: torch.Tensor, recon: torch.Tensor, path: str | Path, limit: int = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(limit, source.shape[0])
    for i in range(n):
        imgs = [
            _with_label(_tensor_to_pil(source[i]), "source"),
            _with_label(_tensor_to_pil(target[i]), "target"),
            _with_label(_tensor_to_pil(noisy[i]), "noisy_target"),
            _with_label(_tensor_to_pil(recon[i]), "x0_hat"),
        ]
        width = sum(img.width for img in imgs)
        height = max(img.height for img in imgs)
        row = Image.new("RGB", (width, height), "#f5f5f5")
        x = 0
        for img in imgs:
            row.paste(img, (x, 0))
            x += img.width
        rows.append(row)

    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def save_val_panel(
    source: torch.Tensor,
    target: torch.Tensor,
    sampled: torch.Tensor,
    high_t_recon: torch.Tensor,
    path: str | Path,
    limit: int = 4,
    high_t_label: str = "x0_hat_high_t",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(limit, source.shape[0])
    for i in range(n):
        imgs = [
            _with_label(_tensor_to_pil(source[i]), "source"),
            _with_label(_tensor_to_pil(target[i]), "target"),
            _with_label(_tensor_to_pil(sampled[i]), "sample"),
            _with_label(_tensor_to_pil(high_t_recon[i]), high_t_label),
        ]
        width = sum(img.width for img in imgs)
        height = max(img.height for img in imgs)
        row = Image.new("RGB", (width, height), "#f5f5f5")
        x = 0
        for img in imgs:
            row.paste(img, (x, 0))
            x += img.width
        rows.append(row)

    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def save_inference_panel(source: torch.Tensor, output: torch.Tensor, path: str | Path, limit: int = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(limit, source.shape[0])
    for i in range(n):
        imgs = [
            _with_label(_tensor_to_pil(source[i]), "source"),
            _with_label(_tensor_to_pil(output[i]), "sample"),
        ]
        width = sum(img.width for img in imgs)
        height = max(img.height for img in imgs)
        row = Image.new("RGB", (width, height), "#f5f5f5")
        x = 0
        for img in imgs:
            row.paste(img, (x, 0))
            x += img.width
        rows.append(row)

    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def save_progress_strip(frames: list[torch.Tensor], path: str | Path, sample_idx: int = 0):
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [_with_label(_tensor_to_pil(frame[sample_idx]), f"step_{i}") for i, frame in enumerate(frames)]
    width = sum(img.width for img in imgs)
    height = max(img.height for img in imgs)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for img in imgs:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(path)
