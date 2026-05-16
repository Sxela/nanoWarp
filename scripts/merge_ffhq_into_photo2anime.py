"""Merge a flat `photo_NNNNNN.png` + `anime_NNNNNN.png` pair directory into
the photo2anime split structure expected by `ProgPairedDataset`.

Creates `data/photo2anime_3k/` (or whatever --out is) with:
    train/source/{existing + ffhq_NNNNNN.png}
    train/target/{existing + ffhq_NNNNNN.png}
    val/source/{existing}        (unchanged)
    val/target/{existing}        (unchanged)

Uses hardlinks where possible (Windows requires the source/target volumes
to match; falls back to copy). Source files in `--source-root` are linked
verbatim — the new combined dir is essentially a view.

Usage:
    python scripts/merge_ffhq_into_photo2anime.py \\
        --base data/photo2anime_1k \\
        --ffhq C:/code/warp/comfywarp/anime_ds_ffhq_1 \\
        --out  data/photo2anime_3k
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Existing photo2anime dataset root with train/{source,target}/ and val/{source,target}/.")
    p.add_argument("--ffhq", required=True,
                   help="Flat dir with photo_NNNNNN.png + anime_NNNNNN.png pairs.")
    p.add_argument("--out", required=True,
                   help="Output dir for the merged dataset.")
    p.add_argument("--prefix", default="ffhq",
                   help="Filename prefix for new pairs (default 'ffhq').")
    p.add_argument("--val-portraits-count", type=int, default=200,
                   help="Take the last N FFHQ pairs and route them to a new "
                        "`val_portraits/` split instead of training. Default 200. "
                        "The existing val/ split is preserved unchanged. Set to 0 "
                        "to disable and route all FFHQ pairs to train.")
    p.add_argument("--skip-base", action="store_true",
                   help="Skip mirroring the --base dataset. Output contains only "
                        "FFHQ pairs (train + val_portraits). Useful for sanity "
                        "tests that isolate FFHQ-domain learning.")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base)
    ffhq = Path(args.ffhq)
    out = Path(args.out)

    # 1. Mirror base splits into out via hardlink (unless --skip-base).
    if args.skip_base:
        print(f"[merge] --skip-base: output will contain only FFHQ pairs")
    else:
        for split in ("train", "val"):
            for side in ("source", "target"):
                src_dir = base / split / side
                dst_dir = out / split / side
                if not src_dir.is_dir():
                    print(f"[warn] missing {src_dir}")
                    continue
                for p in src_dir.iterdir():
                    if p.is_file():
                        _link_or_copy(p, dst_dir / p.name)
        print(f"[merge] mirrored {base} -> {out}")

    # 2. Add FFHQ pairs. Last `val_portraits_count` go to `val_portraits/`,
    #    rest go to `train/`.
    photo_files = sorted(ffhq.glob("photo_*.png"))
    anime_files = sorted(ffhq.glob("anime_*.png"))
    photo_index = {re.search(r"photo_(\d+)", p.stem).group(1): p for p in photo_files
                   if re.search(r"photo_(\d+)", p.stem)}
    # Build the matched-pair list first so we can split deterministically.
    matched = []
    skipped = 0
    for ap in anime_files:
        m = re.search(r"anime_(\d+)", ap.stem)
        if not m:
            continue
        idx = m.group(1)
        sp = photo_index.get(idx)
        if sp is None:
            skipped += 1
            continue
        matched.append((idx, sp, ap))

    val_n = max(0, min(args.val_portraits_count, len(matched)))
    train_pairs = matched[:-val_n] if val_n > 0 else matched
    val_pairs = matched[-val_n:] if val_n > 0 else []

    for idx, sp, ap in train_pairs:
        name = f"{args.prefix}_{idx}.png"
        _link_or_copy(sp, out / "train" / "source" / name)
        _link_or_copy(ap, out / "train" / "target" / name)
    for idx, sp, ap in val_pairs:
        name = f"{args.prefix}_{idx}.png"
        _link_or_copy(sp, out / "val_portraits" / "source" / name)
        _link_or_copy(ap, out / "val_portraits" / "target" / name)
        # When --skip-base there's no legacy val/ to preserve; route the
        # FFHQ val pairs into val/ as well so the training script's required
        # val loader has data to read.
        if args.skip_base:
            _link_or_copy(sp, out / "val" / "source" / name)
            _link_or_copy(ap, out / "val" / "target" / name)
    print(f"[merge] FFHQ pairs: {len(train_pairs)} -> train, {len(val_pairs)} -> val_portraits, "
          f"skipped {skipped} unmatched")

    # 3. Final counts.
    for split in ("train", "val", "val_portraits"):
        sdir = out / split / "source"
        if not sdir.is_dir():
            continue
        n_src = sum(1 for _ in sdir.iterdir())
        n_tgt = sum(1 for _ in (out / split / "target").iterdir())
        print(f"[merge] {split}: source={n_src} target={n_tgt}")


if __name__ == "__main__":
    main()
