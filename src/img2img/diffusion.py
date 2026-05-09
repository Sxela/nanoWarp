from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianImageDiffusion:
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

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x0 = (x_t - sqrt_1mab * eps_hat) / sqrt_ab.clamp(min=1e-8)
        return x0.clamp(0.0, 1.0)

    def training_loss(self, model, source: torch.Tensor, target: torch.Tensor):
        t = torch.randint(0, self.config.timesteps, (target.shape[0],), device=target.device)
        x_t, noise = self.q_sample(target, t)
        eps_hat = model(source, x_t, t)
        loss = F.mse_loss(eps_hat, noise)
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_hat)
        return loss, t, x_t, noise, eps_hat, x0_hat

