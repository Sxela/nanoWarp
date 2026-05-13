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


# Small positive floor for the fractional-pow argument in linear_to_srgb.
# At x = 0, the gradient of x^(1/2.4) is unbounded (1/2.4 < 1, so derivative
# (1/2.4) * x^(1/2.4 - 1) blows up). Even when torch.where selects the linear
# branch, the unselected (high) branch's gradient is computed and multiplied
# by 0; if that gradient is inf or NaN, 0 * NaN = NaN propagates to the
# output. Clamping x to _EPS before the pow keeps the gradient finite.
_EPS = 1e-7


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB to linear conversion (IEC 61966-2-1).

    Input is clamped to [0, 1] at the boundary; out-of-range values get
    safely treated as 0/1. Always computed in fp32 internally for numerical
    stability under bf16/fp16 autocast — `pow` near 0 in bf16 can underflow
    and produce NaN gradients.
    """
    orig_dtype = x.dtype
    x = x.float().clamp(0.0, 1.0)
    cutoff = 0.04045
    low = x / 12.92
    # `(x + 0.055) / 1.055` is always >= 0.052 here, so this pow is safe.
    high = ((x + 0.055) / 1.055) ** 2.4
    out = torch.where(x <= cutoff, low, high)
    return out.to(orig_dtype)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear to sRGB conversion (IEC 61966-2-1).

    Same safeguards as `srgb_to_linear`, plus an `_EPS` floor before the
    fractional pow on the high branch so the gradient stays finite at x=0.
    """
    orig_dtype = x.dtype
    x = x.float().clamp(0.0, 1.0)
    cutoff = 0.0031308
    low = x * 12.92
    # Clamp x to _EPS before the fractional pow so the unselected branch's
    # gradient at x=0 stays finite (see comment on _EPS above).
    x_safe = x.clamp(min=_EPS)
    high = 1.055 * (x_safe ** (1.0 / 2.4)) - 0.055
    out = torch.where(x <= cutoff, low, high)
    return out.to(orig_dtype)
