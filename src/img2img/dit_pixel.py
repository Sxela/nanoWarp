"""Pure-pixel DiT for img2img — HiDream-O1 style, no UNet, no VAE.

Takes (source, noisy_target) pairs in pixel space, patchifies via a single
conv (patch=16 default), runs N DiT-XL-style transformer blocks on the
token grid, unpatchifies the output as a velocity prediction.

Source conditioning: source and noisy_target are concatenated channel-wise
before patchification (6 channels per patch). The model learns to attend
within and across patches to use the source signal.

At default `dim=384, num_layers=11, patch=16`: ~49M params, matches the
49M UNet baseline (exp25) and 51M exp35 budget for a direct A/B.

Per-block accounting (dim=D=384, t_emb_dim=1024, mlp_ratio=4):
  adaLN-zero linear  : t_emb_dim × 6D    = 1024 × 6 × 384 = 2.36M
  MHSA qkv proj      : D × 3D            = 0.44M
  MHSA out proj      : D × D             = 0.15M
  MLP fc1            : D × 4D            = 0.59M
  MLP fc2            : 4D × D            = 0.59M
  Total per block    :                   ≈ 4.13M
  × 11 blocks                            = 45.4M
  + patch_embed (16²·6·D)                = 0.59M
  + time_mlp                             ≈ 2.0M
  + final adaLN + head                   = 1.1M
  Total                                  ≈ 49M  ✓

Token counts:
  256 px  →  16×16 = 256 tokens  (~50 GFLOPs forward @ bs=4)
  512 px  →  32×32 = 1024 tokens (~200 GFLOPs forward @ bs=4)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dit import DiTBlock, sinusoidal_pos_2d
from .model import TimeMLP


class PixelDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: int = 16,
        dim: int = 384,
        num_layers: int = 11,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_dim: int = 256,
        source_in_stem: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.source_in_stem = source_in_stem

        stem_in_ch = in_channels * 2 if source_in_stem else in_channels
        # Patch embedding: a single strided conv produces one token per patch.
        self.patch_embed = nn.Conv2d(stem_in_ch, dim, kernel_size=patch_size, stride=patch_size)

        # Time embedding (matches UNet's TimeMLP output dim → time_dim * 4 features).
        self.time_mlp = TimeMLP(time_dim)
        t_emb_dim = time_dim * 4

        # Transformer stack — adaLN-zero conditioning on t_emb per block.
        self.blocks = nn.ModuleList([
            DiTBlock(dim, t_emb_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

        # Final adaLN-zero head: modulate then project tokens back to a
        # patch-sized chunk of out_channels.
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Linear(t_emb_dim, 2 * dim, bias=True)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        self.head = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.out_channels = out_channels

    def forward(
        self,
        source: torch.Tensor,
        noisy_target: torch.Tensor,
        t: torch.Tensor,
        frame_mask: torch.Tensor | None = None,  # ignored; for API parity
    ) -> torch.Tensor:
        del frame_mask  # interface parity with Img2ImgDiffusionUNet
        if self.source_in_stem:
            x = torch.cat([source, noisy_target], dim=1)
        else:
            x = noisy_target
        B, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H={H}, W={W} not divisible by patch_size={self.patch_size}"
        H_p, W_p = H // self.patch_size, W // self.patch_size

        # Patchify → tokens.
        tok = self.patch_embed(x)                            # (B, dim, H_p, W_p)
        tok = tok.flatten(2).transpose(1, 2)                 # (B, N, dim)

        # 2D sinusoidal positional embedding (size-agnostic).
        pe = sinusoidal_pos_2d(H_p, W_p, self.dim, x.device).to(tok.dtype)
        tok = tok + pe

        # Time embedding.
        t_emb = self.time_mlp(t)

        # Transformer blocks.
        for block in self.blocks:
            tok = block(tok, t_emb)

        # Final adaLN-zero modulation → patch head.
        shift, scale = self.final_modulation(F.silu(t_emb)).chunk(2, dim=-1)
        tok = self.final_norm(tok) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tok = self.head(tok)                                 # (B, N, P*P*C_out)

        # Unpatchify.
        P = self.patch_size
        C = self.out_channels
        out = tok.reshape(B, H_p, W_p, P, P, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_p * P, W_p * P)
        return out
