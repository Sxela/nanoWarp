"""Strip local-machine paths from journey/**/val_metrics_*.json so the
folder is safe to commit. Replaces `checkpoint` field's absolute path with
just the basename (e.g. `C:\\Users\\someuser\\Downloads\\exp35_model.pt`
becomes `exp35_model.pt`).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent / "journey"


def sanitize_value(v: str) -> str:
    """Return bare filename if v looks like a local path, else v unchanged."""
    if not isinstance(v, str):
        return v
    # Match Windows drive paths or POSIX absolute paths containing 'Users' / 'home'.
    if re.search(r"^[A-Za-z]:[\\/]", v) or "/Users/" in v or "/home/" in v:
        # Normalize separators, take basename.
        return v.replace("\\", "/").rsplit("/", 1)[-1]
    return v


def main():
    changed = 0
    for f in ROOT.rglob("val_metrics_*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        dirty = False
        # Sanitize top-level scalar fields.
        for k, v in list(data.items()):
            if isinstance(v, str):
                new = sanitize_value(v)
                if new != v:
                    data[k] = new
                    dirty = True
        if dirty:
            f.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[sanitize] {f.relative_to(ROOT.parent)}")
            changed += 1
    print(f"\n[done] {changed} files sanitized")


if __name__ == "__main__":
    main()
