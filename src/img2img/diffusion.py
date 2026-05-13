from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "eps"  # "eps" or "v"


class GaussianImageDiffusion:
    def __init__(self, config: DiffusionConfig, device: torch.device):
        if config.prediction_type not in ("eps", "v"):
            raise ValueError(f"prediction_type must be 'eps' or 'v', got {config.prediction_type!r}")
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

    def _broadcast(self, vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return vec[t].view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._broadcast(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alpha_bars, t)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    def get_v(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._broadcast(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alpha_bars, t)
        return sqrt_ab * noise - sqrt_1mab * x0

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._broadcast(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alpha_bars, t)
        x0 = (x_t - sqrt_1mab * eps_hat) / sqrt_ab.clamp(min=1e-8)
        return x0.clamp(0.0, 1.0)

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._broadcast(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alpha_bars, t)
        x0 = sqrt_ab * x_t - sqrt_1mab * v_hat
        return x0.clamp(0.0, 1.0)

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._broadcast(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alpha_bars, t)
        return sqrt_1mab * x_t + sqrt_ab * v_hat

    def predict_pair(self, x_t: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor):
        if self.config.prediction_type == "eps":
            return self.predict_x0_from_eps(x_t, t, model_out), model_out
        x0 = self.predict_x0_from_v(x_t, t, model_out)
        eps = self.predict_eps_from_v(x_t, t, model_out)
        return x0, eps

    def _apply_source_dropout(self, source: torch.Tensor, dropout: float) -> torch.Tensor:
        if dropout <= 0:
            return source
        keep = (torch.rand(source.shape[0], 1, 1, 1, device=source.device) > dropout).float()
        return source * keep

    def training_loss(
        self,
        model,
        source: torch.Tensor,
        target: torch.Tensor,
        aux_lpips=None,
        aux_lpips_weight: float = 0.0,
        t_low: int = 0,
        t_high: int | None = None,
        source_dropout: float = 0.0,
    ):
        if t_high is None:
            t_high = self.config.timesteps
        t_low = max(0, min(t_low, self.config.timesteps - 1))
        t_high = max(t_low + 1, min(t_high, self.config.timesteps))
        t = torch.randint(t_low, t_high, (target.shape[0],), device=target.device)
        x_t, noise = self.q_sample(target, t)

        source_in = self._apply_source_dropout(source, source_dropout)
        model_out = model(source_in, x_t, t)

        if self.config.prediction_type == "eps":
            target_for_loss = noise
        else:
            target_for_loss = self.get_v(target, noise, t)

        diffusion_loss = F.mse_loss(model_out, target_for_loss)
        x0_hat, _ = self.predict_pair(x_t, t, model_out)

        lpips_loss = torch.tensor(0.0, device=target.device)
        loss = diffusion_loss
        if aux_lpips is not None and aux_lpips_weight > 0:
            lpips_loss = aux_lpips(x0_hat, target).mean()
            loss = loss + aux_lpips_weight * lpips_loss
        return loss, t, x_t, noise, model_out, x0_hat, diffusion_loss.detach(), lpips_loss.detach()

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
        model_out = model(source, x_t, t)
        if self.config.prediction_type == "v":
            eps_hat = self.predict_eps_from_v(x_t, t, model_out)
        else:
            eps_hat = model_out
        alpha = self._broadcast(self.alphas, t)
        alpha_bar = self._broadcast(self.alpha_bars, t)
        beta = self._broadcast(self.betas, t)

        mean = self._broadcast(self.sqrt_recip_alphas, t) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps_hat)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta) * noise

    @torch.no_grad()
    def ddim_step(self, model, source: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor):
        model_out = model(source, x_t, t)
        x0_hat, eps_hat = self.predict_pair(x_t, t, model_out)

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
