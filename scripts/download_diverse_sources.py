"""Pull diverse real-photo sources for the photo2anime data scale-up
(exp54+ family). Two sources complementary to FFHQ:

1. CelebA-HQ (mattymchen/celeba-hq on HF) — 30k high-res celebrity portraits.
   More pose/expression/age/lighting diversity than FFHQ (which is
   studio-lit Western 25-35yo portraits). Streamed via HuggingFace
   datasets — no full 10GB download required for partial pulls.

2. Places365 small (CSAIL MIT scene dataset, val split) — in-the-wild
   scenes with people. Filtered to a curated list of human-occupied
   scene categories (markets, restaurants, playgrounds, beaches, etc.).
   Source: data.csail.mit.edu/places/places365/val_256.tar (~480MB).

Run on Colab (Linux). Output is square JPGs ready for the local Flux
anime-pair-generation step:

    data/source_pool_diverse/
    ├── celeba_hq/
    │   ├── celeba_000000.jpg
    │   └── manifest.json
    └── places365/
        ├── places_<val_id>.jpg
        └── manifest.json

Smoke (exp54, ~1k pairs after Flux):
    python scripts/download_diverse_sources.py \\
        --celeba-count 500 --places-count 500

Canonical (exp55, ~5k pairs after Flux):
    python scripts/download_diverse_sources.py \\
        --celeba-count 2500 --places-count 2500

Notes:
- CelebA-HQ requires `pip install datasets` (Colab has it by default).
  Streamed: only the requested N samples are pulled, not the full 30k.
- Places365 val is 256x256. We upscale to --width with LANCZOS before
  saving — quality hit but fine for the smoke exp54 hypothesis test.
  For exp55 canonical, consider downloading places train_256.tar (24GB)
  and switching the URL below.
- Idempotent: re-running skips already-downloaded files.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:
    # Tiny fallback so the script runs in barebones envs.
    def tqdm(it=None, total=None, desc=None, **kw):
        if it is None:
            class _Bar:
                def update(self, *a, **k): pass
                def close(self): pass
            print(f"[{desc}] (no tqdm)")
            return _Bar()
        return it

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ===========================================================================
# CelebA-HQ (HF datasets streaming)
# ===========================================================================

# 30k 1024x1024 portraits. Verified via list_datasets search; most-downloaded
# clean mirror of CelebA-HQ on HF in 2026.
CELEBA_HQ_DATASET = "mattymchen/celeba-hq"
CELEBA_HQ_SPLIT = "train"


def pull_celeba_hq(out_dir: Path, target: int, width: int, seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("[celeba] ERROR: 'datasets' not installed. Run: pip install datasets")
        return 0

    print(f"[celeba] streaming {CELEBA_HQ_DATASET} ({CELEBA_HQ_SPLIT}, "
          f"shuffle seed={seed}) ...")
    ds = load_dataset(CELEBA_HQ_DATASET, split=CELEBA_HQ_SPLIT, streaming=True)
    # Streaming-shuffle uses a reservoir buffer so the per-iteration cost is
    # still O(1); buffer_size=4096 is well-distributed without being slow.
    ds = ds.shuffle(seed=seed, buffer_size=4096)

    manifest: list[dict] = []
    bar = tqdm(total=target, desc="[celeba] downloading")
    seen = 0
    for sample in ds:
        if len(manifest) >= target:
            break
        out_path = out_dir / f"celeba_{seen:06d}.jpg"
        seen += 1
        if out_path.exists():
            manifest.append({
                "filename": out_path.name, "source": "celeba_hq",
                "index": seen - 1, "cached": True,
            })
            bar.update(1)
            continue
        try:
            img = sample.get("image")
            if img is None:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            # CelebA-HQ is square 1024x1024; just resize to target width.
            if img.size[0] != width:
                img = img.resize((width, width), Image.LANCZOS)
            img.save(out_path, "JPEG", quality=92)
            manifest.append({
                "filename": out_path.name, "source": "celeba_hq",
                "index": seen - 1,
            })
            bar.update(1)
        except Exception:
            continue
    bar.close()

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[celeba] saved {len(manifest)} images to {out_dir}")
    return len(manifest)


# ===========================================================================
# Places365 (val split, 256x256, ~480MB)
# ===========================================================================

PLACES_VAL_TAR_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
# 5MB tarball containing categories_places365.txt + places365_val.txt +
# places365_train_standard.txt + places365_test.txt — bundled because
# the raw github mirror URLs for these files are unreliable.
PLACES_FILELIST_TAR_URL = (
    "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
)

# Curated Places365 categories where people are usually present in frame.
# Names use the dataset's slash-separated convention (e.g. "bazaar/indoor").
PLACES_PEOPLE_CATEGORIES = [
    "amusement_arcade", "amusement_park", "arena/performance", "auditorium",
    "ballroom", "bar", "bazaar/indoor", "bazaar/outdoor", "beach",
    "beauty_salon", "beer_garden", "beer_hall", "bookstore",
    "bowling_alley", "boxing_ring", "cafeteria", "campsite", "carrousel",
    "church/indoor", "church/outdoor", "classroom",
    "coffee_shop", "computer_room", "conference_center", "conference_room",
    "construction_site", "diner/outdoor", "dining_hall",
    "dining_room", "discotheque", "dressing_room", "fastfood_restaurant",
    "fire_escape", "flea_market/indoor", "food_court",
    "fountain", "gas_station", "general_store/indoor",
    "general_store/outdoor", "gymnasium/indoor", "harbor", "home_office",
    "hospital_room", "ice_cream_parlor", "ice_skating_rink/indoor",
    "ice_skating_rink/outdoor", "industrial_area", "jail_cell",
    "kindergarden_classroom", "kitchen",
    "library/indoor", "library/outdoor", "living_room", "lobby",
    "locker_room", "market/indoor", "market/outdoor",
    "movie_theater/indoor", "museum/indoor", "music_studio", "nursery",
    "nursing_home", "office", "office_cubicles", "operating_room",
    "pavilion", "pharmacy", "playground", "playroom", "plaza",
    "racecourse", "raft", "recreation_room", "restaurant",
    "restaurant_kitchen", "restaurant_patio", "rope_bridge", "schoolhouse",
    "science_museum", "ski_slope", "soccer_field", "stadium/baseball",
    "stadium/football", "stadium/soccer", "stage/indoor", "stage/outdoor",
    "street", "subway_station/platform", "supermarket", "sushi_bar",
    "swimming_pool/indoor", "swimming_pool/outdoor",
    "synagogue/outdoor", "television_room", "television_studio",
    "throne_room",
    "ticket_booth", "toyshop", "train_station/platform", "tree_house",
    "wet_bar", "yard", "youth_hostel",
]


def fetch_places_metadata(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cats = cache_dir / "places_categories.txt"
    labels = cache_dir / "places_val_labels.txt"
    if cats.exists() and labels.exists():
        return cats, labels
    filelist_tar = cache_dir / "filelist_places365-standard.tar"
    if not filelist_tar.exists():
        print(f"[places] fetching filelist tarball (~5MB) ...")
        urllib.request.urlretrieve(PLACES_FILELIST_TAR_URL, filelist_tar)
    print(f"[places] extracting categories + val labels from filelist tar ...")
    with tarfile.open(filelist_tar, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = Path(member.name).name
            target = None
            if base == "categories_places365.txt":
                target = cats
            elif base == "places365_val.txt":
                target = labels
            if target is None:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            target.write_bytes(fh.read())
    if not cats.exists() or not labels.exists():
        raise RuntimeError(
            f"[places] filelist tar did not contain expected files. "
            f"Got cats={cats.exists()} labels={labels.exists()}. "
            f"Tar at {filelist_tar}; inspect manually."
        )
    return cats, labels


def fetch_places_val_tar(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar = cache_dir / "places_val_256.tar"
    if tar.exists() and tar.stat().st_size > 100_000_000:
        return tar
    print(f"[places] downloading val_256.tar (~480MB) to {tar}")
    req = urllib.request.Request(PLACES_VAL_TAR_URL)
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True,
                   desc="[places] download")
        with open(tar, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    return tar


def filter_places(cats_path: Path, labels_path: Path, seed: int) -> list[tuple[str, str]]:
    """Return list of (val_filename, category_name) filtered to people-cats."""
    cat_idx_to_name: dict[int, str] = {}
    name_to_idx: dict[str, int] = {}
    with cats_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, idx = line.rsplit(" ", 1)
            # path is like "/a/airfield" or "/b/bazaar/indoor"
            parts = path.lstrip("/").split("/", 1)
            name = parts[1] if len(parts) > 1 else parts[0]
            cat_idx_to_name[int(idx)] = name
            name_to_idx[name] = int(idx)

    want_idxs: set[int] = set()
    missing: list[str] = []
    for name in PLACES_PEOPLE_CATEGORIES:
        if name in name_to_idx:
            want_idxs.add(name_to_idx[name])
        else:
            missing.append(name)
    if missing:
        print(f"[places] WARNING: {len(missing)} curated categories not in dataset: "
              f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
    print(f"[places] using {len(want_idxs)} people-categories")

    images: list[tuple[str, str]] = []
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, idx = line.rsplit(" ", 1)
            i = int(idx)
            if i in want_idxs:
                images.append((Path(fname).name, cat_idx_to_name[i]))

    print(f"[places] {len(images)} val images in people-categories "
          f"(avg {len(images) / max(len(want_idxs), 1):.1f}/cat)")
    rng = random.Random(seed)
    rng.shuffle(images)
    return images


def extract_places_imgs(tar_path: Path, wanted: list[tuple[str, str]],
                        out_dir: Path, width: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name_to_cat = dict(wanted)
    want_set = set(name_to_cat.keys())

    saved: list[dict] = []
    bar = tqdm(total=len(want_set), desc="[places] extracting")
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = Path(member.name).name
            if base not in want_set:
                continue
            out_path = out_dir / f"places_{Path(base).stem}.jpg"
            if out_path.exists():
                saved.append({
                    "filename": out_path.name, "source": "places365",
                    "original": base, "category": name_to_cat[base],
                    "cached": True,
                })
                bar.update(1)
                continue
            try:
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                data = fh.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                w, h = img.size
                s = min(w, h)
                img = img.crop(((w - s) // 2, (h - s) // 2,
                                (w + s) // 2, (h + s) // 2))
                if img.size[0] != width:
                    img = img.resize((width, width), Image.LANCZOS)
                img.save(out_path, "JPEG", quality=92)
                saved.append({
                    "filename": out_path.name, "source": "places365",
                    "original": base, "category": name_to_cat[base],
                })
                bar.update(1)
            except Exception:
                continue
    bar.close()
    return saved


def pull_places(out_dir: Path, target: int, width: int, seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir.parent / ".cache"
    cats, labels = fetch_places_metadata(cache_dir)
    images = filter_places(cats, labels, seed)
    if not images:
        print("[places] no images matched filter, skipping")
        return 0
    wanted = images[:target]
    tar = fetch_places_val_tar(cache_dir)
    saved = extract_places_imgs(tar, wanted, out_dir, width)
    (out_dir / "manifest.json").write_text(
        json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[places] saved {len(saved)} images to {out_dir}")
    return len(saved)


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/source_pool_diverse", type=Path,
                   help="Output root (will create celeba_hq/ and places365/ subdirs).")
    p.add_argument("--celeba-count", type=int, default=500,
                   help="How many CelebA-HQ portrait photos to pull.")
    p.add_argument("--places-count", type=int, default=500,
                   help="How many Places365 people-scene photos to keep.")
    p.add_argument("--width", type=int, default=512,
                   help="Output image size (square). Matches FFHQ pool resolution.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-celeba", action="store_true")
    p.add_argument("--skip-places", action="store_true")
    args = p.parse_args()

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[diverse] out_dir={out_root} celeba={args.celeba_count} "
          f"places={args.places_count} width={args.width}")

    n_c = 0
    n_p = 0
    if not args.skip_celeba:
        n_c = pull_celeba_hq(
            out_root / "celeba_hq", args.celeba_count, args.width, args.seed,
        )
    if not args.skip_places:
        n_p = pull_places(
            out_root / "places365", args.places_count, args.width, args.seed,
        )

    print()
    print(f"[diverse] done. celeba={n_c}  places={n_p}  total={n_c + n_p}")
    print(f"[diverse] next step: run local Flux to generate anime targets "
          f"from {out_root}/")


if __name__ == "__main__":
    main()
