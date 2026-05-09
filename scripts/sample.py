"""Sampling entrypoint for nanoWarp."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["toy2d", "img2img-v1-val"])
    args, rest = p.parse_known_args()

    stage_file = {
        "toy2d": Path(__file__).resolve().parents[1] / "experiments" / "000_toy_2d" / "sample.py",
        "img2img-v1-val": Path(__file__).resolve().parents[1] / "experiments" / "010_img2img_photo2comics" / "validate.py",
    }[args.stage]

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.argv = [str(stage_file), *rest]
    runpy.run_path(str(stage_file), run_name="__main__")


if __name__ == "__main__":
    main()
