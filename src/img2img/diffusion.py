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
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

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

    def make_sampling_schedule(self, sample_steps: int) -> list[int]:
        sample_steps = max(1, min(sample_steps, self.config.timesteps))
        if sample_steps == self.config.timesteps:
            return list(reversed(range(self.config.timesteps)))
        ts = torch.linspace(self.config.timesteps - 1, 0, sample_steps, device=self.device)
        schedule = ts.round().long().tolist()
        # dedupe while preserving order in case rounding collapses neighbors
        deduped = []
        seen = set()
        for t in schedule:
            if t not in seen:
                deduped.append(int(t))
                seen.add(int(t))
        if deduped[-1] != 0:
            deduped.append(0)
        return deduped

    @torch.no_grad()
    def p_sample(self, model, source: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        eps_hat = model(source, x_t, t)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)

        mean = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps_hat)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta) * noise

    @torch.no_grad()
    def ddim_step(self, model, source: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor):
        eps_hat = model(source, x_t, t)
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_hat)

        next_alpha_bar = torch.ones_like(t, dtype=torch.float32, device=self.device)
        valid = next_t >= 0
        if valid.any():
            next_alpha_bar[valid] = self.alpha_bars[next_t[valid]]
        next_alpha_bar = next_alpha_bar.view(-1, 1, 1, 1)

        x_next = torch.sqrt(next_alpha_bar) * x0_hat + torch.sqrt(1.0 - next_alpha_bar) * eps_hat
        return x_next.clamp(0.0, 1.0), x0_hat

    @torch.no_grad()
    def sample(
        self,
        model,
        source: torch.Tensor,
        image_size: int = 128,
        channels: int = 3,
        sample_steps: int | None = None,
        log_every: int | None = None,
    ):
        b = source.shape[0]
        x = torch.randn(b, channels, image_size, image_size, device=self.device)
        frames: list[torch.Tensor] = []
        if sample_steps is None:
            sample_steps = self.config.timesteps

        if sample_steps >= self.config.timesteps:
            for i in reversed(range(self.config.timesteps)):
                t = torch.full((b,), i, device=self.device, dtype=torch.long)
                x = self.p_sample(model, source, x, t)
                if log_every is not None and (i % log_every == 0 or i == self.config.timesteps - 1 or i == 0):
                    frames.append(x.detach().clamp(0, 1).cpu())
            return x.clamp(0, 1), frames

        schedule = self.make_sampling_schedule(sample_steps)
        for idx, step_t in enumerate(schedule):
            t = torch.full((b,), step_t, device=self.device, dtype=torch.long)
            next_value = schedule[idx + 1] if idx + 1 < len(schedule) else -1
            next_t = torch.full((b,), next_value, device=self.device, dtype=torch.long)
            x, x0_hat = self.ddim_step(model, source, x, t, next_t)
            if log_every is not None and (idx % log_every == 0 or idx == 0 or idx == len(schedule) - 1):
                frames.append(x0_hat.detach().clamp(0, 1).cpu())
        return x.clamp(0, 1), frames
