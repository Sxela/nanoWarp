from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_scatter_png(points: np.ndarray, path: str | Path, size: int = 512, padding: int = 24):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)

    usable = size - 2 * padding
    xy = (points - mins) / span
    xy = xy * usable + padding
    xy[:, 1] = size - xy[:, 1]

    for x, y in xy:
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(30, 30, 30))

    img.save(path)


def save_loss_plot(losses: list[float], path: str | Path, size: int = 512, padding: int = 32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # axes
    draw.line((padding, size - padding, size - padding, size - padding), fill=(180, 180, 180), width=2)
    draw.line((padding, padding, padding, size - padding), fill=(180, 180, 180), width=2)

    if not losses:
        img.save(path)
        return

    vals = np.asarray(losses, dtype=np.float32)
    y_min = float(vals.min())
    y_max = float(vals.max())
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    usable = size - 2 * padding
    pts = []
    for i, loss in enumerate(vals):
        x = padding + usable * (i / max(len(vals) - 1, 1))
        y = size - padding - usable * ((loss - y_min) / (y_max - y_min))
        pts.append((x, y))

    if len(pts) > 1:
        draw.line(pts, fill=(52, 101, 164), width=3)
    else:
        x, y = pts[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(52, 101, 164))

    img.save(path)


def save_image_grid(image_paths: list[str | Path], path: str | Path, thumb_size: int = 192, columns: int = 4, padding: int = 8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.open(p).convert("RGB") for p in image_paths if Path(p).exists()]
    if not imgs:
        return

    rows = (len(imgs) + columns - 1) // columns
    width = columns * thumb_size + (columns + 1) * padding
    height = rows * thumb_size + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), "white")

    for idx, img in enumerate(imgs):
        row = idx // columns
        col = idx % columns
        thumb = img.resize((thumb_size, thumb_size))
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + padding)
        canvas.paste(thumb, (x, y))

    canvas.save(path)
