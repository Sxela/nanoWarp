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

