"""Temporal attention for video-consistent img2img.

HotShot-XL / AnimateDiff style: temporal self-attention inserted after each
spatial feature level, with sinusoidal positional embeddings added to the
normalised sequence before QKV projection (same convention as HotShot-XL's
TemporalAttention — pos info flows into Q, K, and V).

No inter-chunk state. Long-video consistency at inference time uses WAN-style
first-frame conditioning: reinject the last generated frame as the first frame
of the next chunk with mask=0 passed to model.mask_proj.
See model.py (mask_channels, mask_proj) and train_temporal_v2.py for details.

Zero-init gate → pure identity at initialisation; safe to insert into a
pretrained single-frame spatial checkpoint without disturbing outputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _valid_head_dim(channels: int, target: int = 64) -> int:
    """Largest power-of-2 ≤ target that evenly divides channels.

    Flash attention requires head_dim to be a power of 2 (≤256). With
    non-standard model widths (e.g. mc=88 → channels=176, 352, 704) the naive
    num_heads=8 gives non-power-of-2 head_dim. This picks a safe value.
    """
    hd = min(target, channels)
    while hd >= 1:
        if channels % hd == 0 and (hd & (hd - 1)) == 0:
            return hd
        hd -= 1
    return 1


def sinusoidal_pos_emb(T: int, dim: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal position embeddings for T positions. Returns (1, T, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    pos = torch.arange(T, device=device, dtype=torch.float32)
    args = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (T, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(T, 1, device=device)], dim=1)
    return emb.unsqueeze(0)  # (1, T, dim)


class TemporalAttn(nn.Module):
    """Temporal self-attention block (HotShot-XL / AnimateDiff convention).

    Operates on (B*T, C, H, W) feature maps; internally reshapes to
    (B*HW, T, C) for attention across the T frames.

    Usage:
        attn = TemporalAttn(channels)
        attn._num_frames = T   # or via model.set_temporal_frames(T)
        y = attn(x)            # x: (B*T, C, H, W)

    Note: skip 256px levels — B×HW=131072 exceeds CUDA flash-attention kernel
    limits for common batch sizes (head_dim must be a power of 2 and ≥64).
    """

    def __init__(self, channels: int, num_heads: int = 0):
        """num_heads=0 (default): auto-compute from _valid_head_dim(channels)."""
        super().__init__()
        if num_heads == 0:
            head_dim = _valid_head_dim(channels)
            num_heads = channels // head_dim
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        # zero-init gate: identity at init, safe to finetune from spatial ckpt
        self.gate = nn.Parameter(torch.zeros(1))

        self._num_frames: int = 1  # set via model.set_temporal_frames()

    # No-op stubs kept for API compatibility with older training scripts.
    def reset(self) -> None:
        pass

    def detach_kv(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self._num_frames
        if T <= 1:
            return x

        BT, C, H, W = x.shape
        B = BT // T
        HW = H * W
        nh, hd = self.num_heads, self.head_dim

        # (B*T, C, H, W) → (B*HW, T, C)
        h = x.view(B, T, C, HW).permute(0, 3, 1, 2).reshape(B * HW, T, C)

        # Add sinusoidal pos emb to normed sequence before QKV — HotShot-XL style
        pos = sinusoidal_pos_emb(T, C, x.device)  # (1, T, C)
        h_in = self.norm(h) + pos

        qkv = self.to_qkv(h_in).view(B * HW, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B * HW, T, C)
        h = h + self.gate.tanh() * self.out_proj(out)

        return h.view(B, HW, T, C).permute(0, 2, 3, 1).reshape(B * T, C, H, W)
