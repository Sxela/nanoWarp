from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class ToyDiffusion:
    def __init__(self, config: DiffusionConfig, device: torch.device):
        self.config = config
        self.device = device

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_std = torch.sqrt(betas)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    def training_loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.config.timesteps, (x0.shape[0],), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, n: int, data_dim: int = 2) -> torch.Tensor:
        x = torch.randn(n, data_dim, device=self.device)
        for i in reversed(range(self.config.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]
            x = self.sqrt_recip_alphas[i] * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise)
            if i > 0:
                x = x + self.posterior_std[i] * torch.randn_like(x)
        return x

