"""Download a diverse source-image pool for photo->anime pair generation.

Pulls three subsets into `data/source_pool/{ffhq,unsplash,places}/`:
  - FFHQ subset (~5k portraits, 256-512px) via the Hugging Face mirror.
  - Unsplash Lite (~3k chosen from the public CSV).
  - Places365 small subset (~2k scenes) via torchvision.

After download you generate anime pairs with your local Flux edit script,
saving outputs to `data/photo2anime_10k/train/{source,target}/`.

Each subset is independently resumable: re-running skips files already on disk.

Run from repo root:
    PYTHONPATH=. python3 scripts/download_source_pool.py \\
        --out data/source_pool \\
        --ffhq-count 5000 --unsplash-count 3000 --places-count 2000

Dependencies (pip install if missing):
    huggingface_hub, datasets, pandas, requests, tqdm
    torchvision (already installed)
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ---------------------------------------------------------------------------
# FFHQ
# ---------------------------------------------------------------------------

def download_ffhq(out_dir: Path, count: int) -> None:
    """Pull `count` FFHQ images at 512px from the HF mirror.

    Uses datasets streaming so we don't materialise all 70k images.
    Output: out_dir / "NNNNNN.png".
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[ffhq] `pip install datasets` first.")
        return

    # Try a list of known FFHQ mirrors; first one that loads wins.
    # Order: 1024 > 512 > 256 (your Flux edit pipeline benefits from larger inputs).
    candidates = ["pravsels/FFHQ_1024", "Ryan-sjtu/ffhq512-caption", "merkol/ffhq-256"]
    ds = None
    for repo in candidates:
        try:
            print(f"[ffhq] trying {repo} ...")
            ds = load_dataset(repo, split="train", streaming=True)
            print(f"[ffhq] using {repo} (target {count} images -> {out_dir})")
            break
        except Exception as e:
            print(f"[ffhq]   {repo}: {e}")
    if ds is None:
        print("[ffhq] no working mirror found; download manually or pass a different repo")
        return
    saved = 0
    for i, ex in enumerate(tqdm(ds, total=count, desc="ffhq")):
        if saved >= count:
            break
        path = out_dir / f"ffhq_{i:06d}.png"
        if path.exists():
            saved += 1
            continue
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))
        img.convert("RGB").save(path)
        saved += 1
    print(f"[ffhq] saved {saved} images")


# ---------------------------------------------------------------------------
# Unsplash Lite — public dataset, no API key needed.
# ---------------------------------------------------------------------------

# Unsplash Lite release URL. They distribute via unsplash.com/data which 302s
# to a signed S3 URL; the github release asset pattern hasn't been used
# consistently across versions. Use the data-portal URL first; if that fails
# fall back to the github release for the current version.
UNSPLASH_LITE_URLS = [
    "https://unsplash.com/data/lite/1.3.0",
    "https://github.com/unsplash/datasets/releases/download/1.3.0/unsplash-research-dataset-lite-1.3.0.zip",
]


def download_unsplash(out_dir: Path, count: int, topics: list[str]) -> None:
    """Download `count` images from Unsplash Lite.

    The Lite release is a zip of TSV metadata files; one contains photo URLs
    + tags. We grab the TSV, filter by topic, fetch images directly from
    Unsplash's CDN (`download_location` URL).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_zip = out_dir.parent / "unsplash-lite.zip"
    if not cache_zip.exists():
        for url in UNSPLASH_LITE_URLS:
            try:
                print(f"[unsplash] trying {url}")
                with requests.get(url, stream=True, timeout=120,
                                  headers={"User-Agent": "nanoWarp/1.0"}) as r:
                    r.raise_for_status()
                    with open(cache_zip, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                print(f"[unsplash] saved metadata zip -> {cache_zip}")
                break
            except Exception as e:
                print(f"[unsplash]   failed: {e}")
        if not cache_zip.exists():
            print("[unsplash] all URLs failed — skip Unsplash or fetch the zip manually from "
                  "https://unsplash.com/data and place at "
                  f"{cache_zip}")
            return
    import zipfile
    with zipfile.ZipFile(cache_zip) as z:
        names = [n for n in z.namelist() if n.endswith("photos.tsv000")]
        if not names:
            print("[unsplash] photos.tsv000 not found in zip"); return
        with z.open(names[0]) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
            urls: list[tuple[str, str]] = []
            topic_set = {t.lower() for t in topics}
            for row in reader:
                tags = (row.get("photo_tags") or "").lower()
                if topic_set and not any(t in tags for t in topic_set):
                    continue
                url = row.get("photo_image_url")
                pid = row.get("photo_id")
                if url and pid:
                    # use a moderate width (1080px) — Unsplash auto-resizes by URL param
                    urls.append((pid, f"{url}?w=1080&q=80"))
                if len(urls) >= count * 2:  # over-fetch in case some fail
                    break

    print(f"[unsplash] filtered to {len(urls)} candidates -> downloading first {count}")
    saved = 0
    for pid, url in tqdm(urls, total=min(count, len(urls)), desc="unsplash"):
        if saved >= count:
            break
        path = out_dir / f"unsplash_{pid}.jpg"
        if path.exists():
            saved += 1
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(path, "JPEG", quality=90)
            saved += 1
        except Exception as e:
            print(f"[unsplash] skip {pid}: {e}")
        time.sleep(0.05)  # polite rate-limit
    print(f"[unsplash] saved {saved} images")


# ---------------------------------------------------------------------------
# Places365 — small set, torchvision-handled.
# ---------------------------------------------------------------------------

def download_places(out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision.datasets import Places365  # type: ignore
    except ImportError:
        print("[places] torchvision missing")
        return
    cache = out_dir.parent / "places365_cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[places] downloading small Places365 (this is several GB)")
    ds = Places365(root=str(cache), split="val", small=True, download=True)
    n = min(count, len(ds))
    for i in tqdm(range(n), desc="places"):
        img, _label = ds[i]
        path = out_dir / f"places_{i:06d}.jpg"
        if path.exists():
            continue
        img.convert("RGB").save(path, "JPEG", quality=92)
    print(f"[places] saved {n} images")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/source_pool")
    p.add_argument("--ffhq-count", type=int, default=5000)
    p.add_argument("--unsplash-count", type=int, default=3000)
    p.add_argument("--places-count", type=int, default=2000)
    p.add_argument("--unsplash-topics", nargs="*",
                   default=["people", "portrait", "face"],
                   help="Tag-based filters for Unsplash Lite (case-insensitive).")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["ffhq", "unsplash", "places"],
                   help="Sources to skip (e.g. --skip places to avoid the multi-GB download).")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if "ffhq" not in args.skip:
        download_ffhq(root / "ffhq", args.ffhq_count)
    if "unsplash" not in args.skip:
        download_unsplash(root / "unsplash", args.unsplash_count, args.unsplash_topics)
    if "places" not in args.skip:
        download_places(root / "places", args.places_count)
    print("done.")


if __name__ == "__main__":
    main()
