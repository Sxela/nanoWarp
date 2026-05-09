from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def apply_yaml_config(parser: argparse.ArgumentParser):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()
    if known.config:
        cfg_path = Path(known.config)
        data = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {cfg_path}")
        parser.set_defaults(**data)
    parser.add_argument("--config", type=str, default=known.config)
    return parser

