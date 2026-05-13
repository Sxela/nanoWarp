"""Materialize the photo2anime paired dataset into the layout nanoWarp expects.

Source images are flat in `<src_root>/photo_NNNNNN.png` and `<src_root>/anime_NNNNNN.png`.
This script copies them into:

    <out_root>/train/source/NNNNNN.png
    <out_root>/train/target/NNNNNN.png
    <out_root>/val/source/NNNNNN.png
    <out_root>/val/target/NNNNNN.png

The split is a deterministic tail split: the last `--val-count` indices go to val.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PHOTO_RE = re.compile(r"^photo_(\d{6})\.png$", re.IGNORECASE)
ANIME_RE = re.compile(r"^anime_(\d{6})\.png$", re.IGNORECASE)


def collect_indices(src_root: Path) -> list[str]:
    photos: dict[str, Path] = {}
    animes: dict[str, Path] = {}
    for p in sorted(src_root.iterdir()):
        if not p.is_file():
            continue
        m = PHOTO_RE.match(p.name)
        if m:
            photos[m.group(1)] = p
            continue
        m = ANIME_RE.match(p.name)
        if m:
            animes[m.group(1)] = p
    keys = sorted(set(photos) & set(animes))
    only_photo = sorted(set(photos) - set(animes))
    only_anime = sorted(set(animes) - set(photos))
    if only_photo:
        print(f"[warn] {len(only_photo)} photos without an anime pair (e.g. {only_photo[:3]})")
    if only_anime:
        print(f"[warn] {len(only_anime)} anime without a photo pair (e.g. {only_anime[:3]})")
    return [(k, photos[k], animes[k]) for k in keys]


def copy_pair(src: Path, tgt_dir: Path, key: str) -> None:
    tgt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tgt_dir / f"{key}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=r"C:\code\warp\comfywarp\anime_ds")
    p.add_argument("--out", default=r"c:\code\warp\nanoWarp\data\photo2anime")
    p.add_argument("--val-count", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    if not src_root.exists():
        raise SystemExit(f"src not found: {src_root}")

    pairs = collect_indices(src_root)
    if not pairs:
        raise SystemExit(f"no paired files found under {src_root}")
    if args.val_count >= len(pairs):
        raise SystemExit(f"val_count {args.val_count} >= total pairs {len(pairs)}")

    train_pairs = pairs[: -args.val_count]
    val_pairs = pairs[-args.val_count :]
    print(f"total pairs: {len(pairs)}")
    print(f"train: {len(train_pairs)}  ({train_pairs[0][0]}..{train_pairs[-1][0]})")
    print(f"val:   {len(val_pairs)}   ({val_pairs[0][0]}..{val_pairs[-1][0]})")
    print(f"out:   {out_root}")

    if args.dry_run:
        print("[dry-run] no files copied")
        return

    for split, items in (("train", train_pairs), ("val", val_pairs)):
        src_dir = out_root / split / "source"
        tgt_dir = out_root / split / "target"
        src_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for key, photo_path, anime_path in items:
            copy_pair(photo_path, src_dir, key)
            copy_pair(anime_path, tgt_dir, key)
        print(f"[{split}] copied {len(items)} pairs")

    print("done")


if __name__ == "__main__":
    main()
