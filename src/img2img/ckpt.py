"""Centralized checkpoint loader.

Reconstructs the model from a checkpoint by preferring fields in
`ckpt["config"]` (saved CLI args from train_exp32_prog512.py) and falling
back to state_dict shape detection only for legacy checkpoints that
predate a given config field.

Usage:
    from src.img2img.ckpt import build_model_from_ckpt
    model, state_dict, cfg = build_model_from_ckpt(ckpt_path, device, use_ema=True)
"""

from __future__ import annotations

from typing import Any

import torch

from .model import Img2ImgDiffusionUNet
from .dit_pixel import PixelDiT


def build_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    use_ema: bool = True,
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    """Load a checkpoint and reconstruct the model.

    Args:
        ckpt_path: path to a *.pt saved by train_exp32_prog512.py.
        device: target device.
        use_ema: if True and ckpt contains "ema_model", load EMA weights.
        verbose: print one-line summary of detected/configured arch.

    Returns:
        (model, state_dict, train_cfg) — model is on `device` and in eval mode,
        with weights loaded. state_dict is the raw dict used (for downstream
        introspection). train_cfg is the saved CLI-args dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    state_key = "ema_model" if (use_ema and "ema_model" in ckpt) else "model"
    sd = ckpt[state_key]

    # Architecture dispatch: prefer config; fall back to state_dict signature.
    arch = cfg.get("arch")
    if arch is None:
        arch = "pixel_dit" if "patch_embed.weight" in sd else "unet"

    if arch == "pixel_dit":
        model = _build_pixel_dit(cfg, sd, device, verbose)
    else:
        model = _build_unet(cfg, sd, device, verbose)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, sd, cfg


def _build_pixel_dit(cfg, sd, device, verbose):
    # Shape detection from state_dict — patch_embed.weight is the canonical
    # carrier of (dim, in_ch, patch, patch).
    patch_w = sd["patch_embed.weight"]
    dim = int(cfg.get("dit_pixel_dim", patch_w.shape[0]))
    patch_size = int(cfg.get("dit_pixel_patch", patch_w.shape[-1]))
    num_layers = int(cfg.get("dit_pixel_layers") or max(
        int(k.split(".")[1]) + 1 for k in sd.keys() if k.startswith("blocks.")
    ))
    num_heads = max(1, dim // 64)
    if verbose:
        print(f"[ckpt] pixel_dit  dim={dim}  layers={num_layers}  patch={patch_size}")
    return PixelDiT(
        in_channels=3, out_channels=3,
        patch_size=patch_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,
        source_in_stem=True,
    ).to(device)


def _build_unet(cfg, sd, device, verbose):
    attn_res = tuple(int(x) for x in str(cfg.get("attn_resolutions", "8")).split(",") if x.strip())
    color_space = cfg.get("color_space", "srgb")

    # Architecture metadata: prefer config, fall back to state_dict shapes.
    if "source_in_stem" in cfg:
        source_in_stem = bool(cfg["source_in_stem"])
    else:
        source_in_stem = sd["in_conv.weight"].shape[1] == 6
    if "no_source_encoder" in cfg:
        use_source_encoder = not cfg["no_source_encoder"]
    else:
        use_source_encoder = any(k.startswith("source_encoder.") for k in sd.keys())
    upsample_type = cfg.get("upsample_type", "resize_conv")

    # image_size determines which attn modules get instantiated. For exp32+
    # the model is built at 512 regardless of training resolution.
    if "image_size" in cfg:
        image_size = int(cfg["image_size"])
    else:
        present = {i for i in range(1, 5) if f"attn{i}.norm.weight" in sd}
        attn_set = set(attn_res)
        image_size = next(
            (sz for sz in (128, 256, 512, 1024)
             if {i for i in range(1, 5) if (sz >> (i - 1)) in attn_set} == present),
            128,
        )

    # Optional modules — state_dict is canonical (presence means True).
    use_decoder_attn = any(k.startswith("attn_dec") for k in sd.keys())
    use_source_pyramid = any(k.startswith("source_pyramid.") for k in sd.keys())
    use_dit_bottleneck = any(k.startswith("dit_bottleneck.") for k in sd.keys())
    use_cross_attn_cond = any(k.startswith("cross_attn_dec4.") for k in sd.keys())
    num_dit_blocks = max(
        (int(k.split(".")[2]) + 1 for k in sd.keys()
         if k.startswith("dit_bottleneck.blocks.")),
        default=int(cfg.get("num_dit_blocks", 4)),
    )

    if verbose:
        print(f"[ckpt] unet  mc={cfg.get('model_ch', 64)}  attn_res={attn_res}  "
              f"image_size={image_size}  source_in_stem={source_in_stem}  "
              f"use_source_encoder={use_source_encoder}  "
              f"decoder_attn={use_decoder_attn}  pyramid={use_source_pyramid}  "
              f"cross_attn={use_cross_attn_cond}  "
              f"dit={use_dit_bottleneck}({num_dit_blocks} blk)")

    return Img2ImgDiffusionUNet(
        model_ch=cfg.get("model_ch", 64),
        pretrained_source_encoder=False,
        source_in_stem=source_in_stem,
        use_source_encoder=use_source_encoder,
        upsample_type=upsample_type,
        attn_resolutions=attn_res,
        image_size=image_size,
        color_space=color_space,
        use_decoder_attn=use_decoder_attn,
        use_source_pyramid=use_source_pyramid,
        use_dit_bottleneck=use_dit_bottleneck,
        num_dit_blocks=num_dit_blocks,
        use_cross_attn_cond=use_cross_attn_cond,
    ).to(device)
