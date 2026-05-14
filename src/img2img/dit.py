"""DiT bottleneck for Img2ImgDiffusionUNet (exp36).

Replaces the original convolutional bottleneck (mid_attn + mid2 ResBlock) with
a stack of DiT-XL-style transformer blocks operating on the flattened
(H/16 × W/16, cm) token grid. Time conditioning enters via adaLN-zero so the
stack is identity at init and a no-DiT checkpoint loads cleanly via
strict=False.

The channel-widening ResBlock `mid1` (c4 → cm) is preserved upstream so the
DiT stack always operates at constant width cm.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _valid_head_dim(channels: int, target: int = 64) -> int:
    """Largest power-of-2 ≤ target that evenly divides channels.

    Flash attention requires head_dim to be a power of 2 (≤256). At
    channels=704 (cm at mc=88), this picks head_dim=64 → num_heads=11.
    """
    hd = min(target, channels)
    while hd >= 1:
        if channels % hd == 0 and (hd & (hd - 1)) == 0:
            return hd
        hd -= 1
    return 1


def sinusoidal_pos_2d(H: int, W: int, dim: int, device: torch.device) -> torch.Tensor:
    """2D sinusoidal positional embeddings flattened to a sequence.

    Returns (1, H*W, dim). Half the channels encode the y-coordinate, the
    other half encode the x-coordinate. Size-agnostic so the same DiT stack
    can be evaluated at multiple bottleneck resolutions (128/256/512px input
    → 8×8 / 16×16 / 32×32 tokens) without learned parameters.
    """
    half = dim // 2
    half = half - (half % 2)  # ensure even split per axis
    quarter = half // 2

    def axis_pe(n: int, d: int) -> torch.Tensor:
        # 1D sinusoidal PE of length n, dim d (d must be even).
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(d // 2, device=device, dtype=torch.float32)
            / max(d // 2 - 1, 1)
        )
        pos = torch.arange(n, device=device, dtype=torch.float32)
        args = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (n, d//2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (n, d)

    pe_y = axis_pe(H, half)  # (H, half)
    pe_x = axis_pe(W, half)  # (W, half)
    # Broadcast to (H, W, dim): y in first half, x in second half.
    pe = torch.zeros(H, W, dim, device=device)
    pe[:, :, :half] = pe_y.unsqueeze(1)               # (H, 1, half) → (H, W, half)
    pe[:, :, half : 2 * half] = pe_x.unsqueeze(0)     # (1, W, half) → (H, W, half)
    return pe.reshape(1, H * W, dim)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FiLM-style affine on the sequence axis. x: (B, N, D); shift/scale: (B, D)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Standard DiT block with adaLN-zero conditioning.

        norm1(x) → modulate → MHSA → gated residual
        norm2(x) → modulate → MLP  → gated residual

    The (shift, scale, gate) × 2 modulation values are produced by a single
    Linear(t_emb_dim → 6 D) initialised to zero, so the block emits its input
    unchanged at init.
    """

    def __init__(
        self,
        dim: int,
        t_emb_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_out = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )

        # adaLN-zero: a single Linear produces (shift, scale, gate) × 2 from t_emb.
        self.ada_ln = nn.Linear(t_emb_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.ada_ln.weight)
        nn.init.zeros_(self.ada_ln.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D); t_emb: (B, t_emb_dim) — already SiLU'd upstream
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln(F.silu(t_emb)).chunk(6, dim=-1)
        )

        # Self-attention
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        B, N, D = h.shape
        nh = self.num_heads
        hd = D // nh
        qkv = self.qkv(h).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # each (B, nh, N, hd)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        x = x + gate_msa.unsqueeze(1) * self.attn_out(attn)

        # MLP
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiTBottleneck(nn.Module):
    """Stack of DiT blocks operating on a (B, C, H, W) feature map.

    Flattens to (B, H*W, C), adds 2D sinusoidal positional embeddings, runs
    `num_blocks` DiT blocks with adaLN-zero conditioning on `t_emb`, and
    reshapes back to (B, C, H, W).
    """

    def __init__(
        self,
        dim: int,
        t_emb_dim: int,
        num_blocks: int = 4,
        num_heads: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if num_heads == 0:
            head_dim = _valid_head_dim(dim)
            num_heads = dim // head_dim
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.blocks = nn.ModuleList([
            DiTBlock(dim, t_emb_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Spatial → sequence
        h = x.flatten(2).transpose(1, 2)                        # (B, H*W, C)
        h = h + sinusoidal_pos_2d(H, W, C, x.device).to(h.dtype)
        for block in self.blocks:
            h = block(h, t_emb)
        # Sequence → spatial
        return h.transpose(1, 2).reshape(B, C, H, W)
