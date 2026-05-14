"""Decoder LoRA for img2img UNet (exp29).

Injects rank-r low-rank adapters on the conv1 + conv2 of dec4/dec3/dec2/dec1
ResBlocks. The base (frozen) conv weight is untouched; a lightweight
delta_W = B @ A is added per forward pass.

Zero-init B → identity at init (output identical to frozen backbone until
training begins).

Usage:
    from src.img2img.decoder_lora import add_decoder_lora, decoder_lora_params

    add_decoder_lora(model, rank=8)       # patches model in-place
    params = decoder_lora_params(model)   # returns list of trainable LoRA params
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from src.img2img.model import Img2ImgDiffusionUNet


class LoRAConv2d(nn.Module):
    """Frozen nn.Conv2d + trainable rank-r delta weight.

    delta_W = lora_B @ lora_A  (shape: out_ch × in_ch × kH × kW)
    output  = base_conv(x) + F.conv2d(x, delta_W, ...)

    B is zero-init → delta_W = 0 at init → identical to base_conv.
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 8):
        super().__init__()
        out_ch, in_ch, kH, kW = conv.weight.shape
        self.conv = conv  # kept frozen by caller

        fan_in = in_ch * kH * kW
        self.lora_A = nn.Parameter(torch.empty(rank, fan_in))
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank))

        # Kaiming-uniform init for A (same as default Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self._stride = conv.stride
        self._padding = conv.padding
        self._dilation = conv.dilation
        self._groups = conv.groups
        self._kH = kH
        self._kW = kW
        self._in_ch = in_ch
        self._out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        # delta_W: (out_ch, in_ch, kH, kW)
        delta_w = (self.lora_B @ self.lora_A).view(
            self._out_ch, self._in_ch, self._kH, self._kW
        )
        return base + F.conv2d(x, delta_w, None,
                               self._stride, self._padding,
                               self._dilation, self._groups)

    def lora_params(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def add_decoder_lora(model: "Img2ImgDiffusionUNet", rank: int = 8) -> None:
    """Patch dec4/dec3/dec2/dec1 ResBlock conv1+conv2 in-place with LoRAConv2d.

    The base conv weights are left frozen (requires_grad=False on conv.weight/bias).
    Only lora_A and lora_B are trainable.
    """
    for block_name in ("dec4", "dec3", "dec2", "dec1"):
        block = getattr(model, block_name)
        for conv_name in ("conv1", "conv2"):
            conv = getattr(block, conv_name)
            assert isinstance(conv, nn.Conv2d), \
                f"{block_name}.{conv_name} is not Conv2d: {type(conv)}"
            # Freeze base conv
            conv.weight.requires_grad_(False)
            if conv.bias is not None:
                conv.bias.requires_grad_(False)
            # Replace with LoRA wrapper
            setattr(block, conv_name, LoRAConv2d(conv, rank=rank).to(conv.weight.device))


def decoder_lora_params(model: "Img2ImgDiffusionUNet") -> list[nn.Parameter]:
    """Return all LoRA parameters from the patched decoder blocks."""
    params = []
    for block_name in ("dec4", "dec3", "dec2", "dec1"):
        block = getattr(model, block_name)
        for conv_name in ("conv1", "conv2"):
            conv = getattr(block, conv_name)
            if isinstance(conv, LoRAConv2d):
                params.extend(conv.lora_params())
    return params
