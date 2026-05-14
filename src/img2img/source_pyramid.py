"""Source feature pyramid + FiLM modulation for Img2ImgDiffusionUNet.

In-model alternative to the optional ResNet18 source encoder. Stays inside the
single-checkpoint deployment story (no external pretrained backbone at
inference time).

Pyramid: a tiny 4-stage conv stack run once on the raw source per forward
pass, producing features at the four decoder resolutions (matching UNet
widths c1, c2, c3, c4).

FiLM: per-level 1x1 conv produces (γ, β) from a pyramid feature; decoder
activation becomes x * (1 + γ) + β. Both γ and β are zero-init → identity
at init → safe insertion: a no-pyramid checkpoint loads cleanly with
strict=False, and a pyramid-enabled model at step 0 outputs the same values
as a no-pyramid model with the same backbone weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PyramidStage(nn.Module):
    """One downsampling stage: GroupNorm → SiLU → Conv3x3 → AvgPool2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.down = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.conv(F.silu(self.norm(x))))


class SourcePyramid(nn.Module):
    """4-resolution source feature pyramid.

    Input:  (B, 3, H, W) source image in [0, 1].
    Output: list of 4 tensors at resolutions H, H/2, H/4, H/8 with channels
            (c1, c2, c3, c4) respectively.

    Param count at default widths (c1=88, c2=176, c3=352, c4=352): ~1.8M.
    """

    def __init__(self, channels: tuple[int, int, int, int] = (88, 176, 352, 352)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Conv2d(3, c1, 3, padding=1)            # (c1, H, W)
        self.stage1 = PyramidStage(c1, c2)                    # (c2, H/2, W/2)
        self.stage2 = PyramidStage(c2, c3)                    # (c3, H/4, W/4)
        self.stage3 = PyramidStage(c3, c4)                    # (c4, H/8, W/8)

    def forward(self, source: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(source)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f0, f1, f2, f3]


class FiLM(nn.Module):
    """Feature-wise linear modulation.

    Produces (γ, β) of shape (B, target_ch, H, W) from a same-resolution
    condition tensor of shape (B, cond_ch, H, W), then returns
        x * (1 + γ) + β
    where x is (B, target_ch, H, W). Zero-init both γ and β → identity output
    at init, so the host model can adopt FiLM without disrupting pretrained
    weights.
    """

    def __init__(self, cond_ch: int, target_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(cond_ch, target_ch * 2, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=1)
        return x * (1.0 + scale) + shift
