"""Build a per-experiment `journey/expNN/` folder from available artifacts.

For each experiment number found in the captain's logs, creates a folder
containing:
  - description.md      — captain's log section + recipe summary
  - run_script.sh       — the run_expNN_*.sh from scripts/ if present
  - val_metrics.json    — final val metrics if available (out/val_exp*)
  - panels/*.png        — training panels (from out/ and Downloads/)
  - checkpoint.pt       — hardlink to local ckpt if found (Downloads/)

Idempotent: re-running picks up newly-added artifacts and overwrites
description.md with the latest captain's log content.

Usage:
    python scripts/build_journey.py            # all experiments
    python scripts/build_journey.py 1 7 25 35  # specific exps
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
SCRIPTS = REPO / "scripts"
JOURNEY = REPO / "journey"
DOWNLOADS = Path("C:/Users/defil/Downloads")
LOGS = [REPO / "docs" / "captains_log.md", REPO / "docs" / "captains_log_3k.md"]


def parse_logs() -> dict[str, str]:
    """Return {expNN_tag: section_markdown} extracted from both captain's logs.

    Section headers like `## exp35 — ...` or `### exp08-lpips — ...`. Captures
    everything until the next `##` / `###` header.
    """
    sections: dict[str, str] = {}
    for log_path in LOGS:
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"^(#{2,3})\s+(exp[0-9a-z_/]+)\b[^\n]*\n",
            re.MULTILINE | re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            tag = m.group(2).lower()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].rstrip() + "\n"
            # If the tag appears in multiple logs / sections, keep the longest entry.
            if tag in sections and len(sections[tag]) >= len(body):
                continue
            sections[tag] = body
    return sections


def normalize_exp_num(tag: str) -> str | None:
    """Map an exp-tag like `exp08-lpips`, `exp07b`, `exp14v2` to `08`, `07b`, `14v2`.

    Returns the canonical 2-digit-with-suffix form, or None if unparseable.
    """
    m = re.match(r"^exp(\d{1,3})([a-z_v0-9]*)", tag.lower())
    if not m:
        return None
    num = int(m.group(1))
    suffix = m.group(2)
    # Strip "lpips" / "noenc" / etc. suffixes — they're stage tags, not exp numbers.
    # Keep only short letter suffixes like b, c, v2.
    if suffix and not re.fullmatch(r"[a-z]\d?|v\d", suffix):
        suffix = ""
    return f"{num:02d}{suffix}"


def find_outdir(num: str) -> list[Path]:
    """Find all out/expNN_* directories matching this exp number."""
    # Accept both 2-digit and 1-digit prefixes for back-compat.
    candidates = []
    for prefix in (f"exp{num}", f"exp{int(num) if num.isdigit() else num}",
                   f"exp{num.rstrip('abcv0123456789')}"):
        candidates.extend(OUT.glob(f"{prefix}_*"))
        candidates.extend(OUT.glob(f"{prefix}/"))
    return sorted(set(p for p in candidates if p.is_dir()))


def find_val_dirs(num: str) -> list[Path]:
    """Find all out/val_*expNN* dirs with metrics for this exp."""
    candidates = []
    candidates.extend(OUT.glob(f"val_exp{num}_*"))
    candidates.extend(OUT.glob(f"val_exp{num.rstrip('abcv0123456789')}_*"))
    candidates.extend(OUT.glob(f"val_e{num.rstrip('abcv0123456789')}_*"))  # legacy e25-style
    return sorted(set(p for p in candidates if p.is_dir()))


def find_run_script(num: str) -> Path | None:
    """Find scripts/run_expNN_*.sh."""
    for p in SCRIPTS.glob(f"run_exp{num}_*.sh"):
        return p
    for p in SCRIPTS.glob(f"run_exp{int(num) if num.isdigit() else num}_*.sh"):
        return p
    return None


def find_panels_in_dir(dir_path: Path, exp_tag: str | None = None, limit: int = 6) -> list[Path]:
    """Return up to `limit` panel PNG files from a directory."""
    if not dir_path.exists():
        return []
    pngs = list(dir_path.glob("*.png"))
    # Sort by mtime so latest panels come last.
    pngs.sort(key=lambda p: p.stat().st_mtime)
    return pngs[-limit:]


def find_downloads_panels(num: str, limit: int = 6) -> list[Path]:
    """Find panels in Downloads/ that look like they belong to this exp."""
    if not DOWNLOADS.exists():
        return []
    candidates = []
    # Try a few naming conventions seen in the wild.
    patterns = [
        f"exp{num}_*.png",                # exp50_panel_step_*.png
        f"exp{num} *.png",                # exp 11 30k val_panel_000.png
        f"exp {num} *.png",
        f"e{num.lstrip('0')}*.png",       # e25 panel_step_*.png
        f"e{num}*.png",
        f"exp{num.rstrip('abcv0123456789')}*.png",
        f"ep{num.lstrip('0')}*.png",
    ]
    seen = set()
    for pat in patterns:
        for p in DOWNLOADS.glob(pat):
            if p.is_file() and p.suffix.lower() == ".png" and p not in seen:
                seen.add(p)
                candidates.append(p)
    candidates.sort()
    return candidates[:limit]


def find_checkpoint(num: str) -> Path | None:
    """Find a .pt checkpoint in Downloads/ for this exp number."""
    if not DOWNLOADS.exists():
        return None
    # Different historical naming conventions.
    patterns = [
        f"exp{num}_model.pt",
        f"exp{num}-80k-model.pt",
        f"exp {num} *model*.pt",
        f"e{num.lstrip('0')}-model*.pt",
        f"exp{num}*.pt",
        f"exp {num} *.pt",
        f"e{num}*model*.pt",
    ]
    for pat in patterns:
        for p in DOWNLOADS.glob(pat):
            if p.is_file() and p.suffix == ".pt":
                return p
    return None


def copy_file(src: Path, dst: Path):
    """Copy a file, prefer hardlink for size."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def build_for(num: str, section: str | None, dry_run: bool = False) -> None:
    """Build journey/expNN/ from available artifacts."""
    target = JOURNEY / f"exp{num}"
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[journey] exp{num} -> {target}")

    # 1. Description (captain's log section).
    desc_path = target / "description.md"
    if section:
        desc_path.write_text(section, encoding="utf-8")
        print(f"  description.md ({len(section)} bytes)")
    elif not desc_path.exists():
        desc_path.write_text(f"# exp{num}\n\n(no captain's log entry found)\n", encoding="utf-8")
        print(f"  description.md (empty placeholder)")

    # 2. Run script.
    script_path = find_run_script(num)
    if script_path:
        copy_file(script_path, target / "run_script.sh")
        print(f"  run_script.sh <- {script_path.name}")

    # 3. Val metrics (newest).
    val_dirs = find_val_dirs(num)
    for val_dir in val_dirs:
        metrics = val_dir / "val_metrics.json"
        if metrics.exists():
            tag = val_dir.name.replace(f"val_exp{num}_", "").replace("val_", "")
            tag = tag if tag else "final"
            copy_file(metrics, target / f"val_metrics_{tag}.json")
            print(f"  val_metrics_{tag}.json <- {val_dir.name}")

    # 4. Panels (from out/ and Downloads/).
    panels_dir = target / "panels"
    panels_added = 0
    for outdir in find_outdir(num):
        for p in find_panels_in_dir(outdir, limit=4):
            copy_file(p, panels_dir / p.name)
            panels_added += 1
    for p in find_downloads_panels(num, limit=8):
        # Sanitize filename for cross-platform.
        safe_name = p.name.replace(" ", "_")
        copy_file(p, panels_dir / safe_name)
        panels_added += 1
    if panels_added:
        print(f"  panels/ ({panels_added} files)")

    # 5. Checkpoint.
    ckpt = find_checkpoint(num)
    if ckpt:
        copy_file(ckpt, target / "checkpoint.pt")
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"  checkpoint.pt <- {ckpt.name} ({size_mb:.0f}MB hardlink)")


def main():
    args = sys.argv[1:]
    sections = parse_logs()
    print(f"[parse] {len(sections)} captain's-log sections")

    # Build the set of exp numbers we have any info for.
    all_nums = set()
    for tag in sections.keys():
        n = normalize_exp_num(tag)
        if n:
            all_nums.add(n)
    # Also pick up exps that have a run script but no log entry.
    for p in SCRIPTS.glob("run_exp*.sh"):
        m = re.match(r"run_exp(\d{1,3}[a-z]*)_", p.name)
        if m:
            n = m.group(1).zfill(2) if m.group(1).isdigit() else m.group(1)
            all_nums.add(n if len(n) >= 2 else n.zfill(2))

    if args:
        targets = [a.zfill(2) if a.isdigit() else a for a in args]
    else:
        targets = sorted(all_nums)

    for num in targets:
        # Find the matching captain's-log section. Prefer plain `expNN`, fall
        # back to any `expNN<suffix>` entry.
        section = None
        for tag, body in sections.items():
            ntag = normalize_exp_num(tag)
            if ntag == num:
                section = body
                break
        # Loose match: log uses tag with suffix.
        if section is None:
            for tag, body in sections.items():
                if normalize_exp_num(tag) and normalize_exp_num(tag).startswith(num):
                    section = body
                    break
        build_for(num, section)


if __name__ == "__main__":
    main()
