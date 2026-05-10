"""sRGB <-> linear RGB conversions.

PIL loads PNG/JPG bytes as gamma-encoded sRGB in [0, 1]. Linear-light
operations (Gaussian noise, linear blends like the FM interpolant
`(1-t)*source + t*target`, bilinear upsampling) are only mathematically
correct in linear space. ImageNet-pretrained networks (ResNet18 source
encoder, LPIPS SqueezeNet) were trained on sRGB and expect sRGB input.

So when training in linear, we apply these conversions only at the
boundaries: dataset loader (sRGB -> linear after PIL aug), source encoder
input (linear -> sRGB), LPIPS input (linear -> sRGB), and panel save
(linear -> sRGB for display).
"""

from __future__ import annotations

import torch


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Standard IEC 61966-2-1 sRGB to linear conversion. Input/output in [0, 1]."""
    cutoff = 0.04045
    low = x / 12.92
    high = ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4
    return torch.where(x <= cutoff, low, high)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear to sRGB conversion. Input/output in [0, 1]."""
    cutoff = 0.0031308
    low = x * 12.92
    high = 1.055 * (x.clamp(min=0) ** (1.0 / 2.4)) - 0.055
    return torch.where(x <= cutoff, low, high)
