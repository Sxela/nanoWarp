"""Data preparation/debug entrypoint for nanoWarp."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["paired-img2img-debug"])
    args, rest = p.parse_known_args()

    stage_file = {
        "paired-img2img-debug": Path(__file__).resolve().parents[1] / "scripts" / "debug_paired_dataset.py",
    }[args.stage]

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.argv = [str(stage_file), *rest]
    runpy.run_path(str(stage_file), run_name="__main__")


if __name__ == "__main__":
    main()
