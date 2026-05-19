"""Direct image-to-image rectified flow matching.

Path:    x_t = (1 - t) * source + t * target + sigma * noise         for t in [0, 1]
Target:  v_target = target - source                                   (constant velocity along the line)
Sample:  start at x = source, integrate dx/dt = model(source, x, t)   from t = 0 to t = 1

Compared to GaussianImageDiffusion:
- No alpha/beta schedule, no clamp gymnastics, no eps/v parameterization split.
- Inference starts from the source image directly, never from pure Gaussian noise.
- The model output is interpreted as a per-pixel velocity in image-space.

The interface is shaped to be drop-in compatible with GaussianImageDiffusion so the
trainer can switch between methods via a flag:

    training_loss(model, source, target, ...) -> (loss, t_display, x_t, noise, model_out,
                                                  x_target_hat, primary_loss_detached, lpips_loss_detached)
    sample(model, source, image_size, channels, sample_steps, log_every) -> (samples, frames)

The `t_low` / `t_high` arguments in `training_loss` are kept on the integer-timestep scale
so that callers can share warmup logic across methods. They are converted internally to the
[0, 1] continuous flow range. For flow matching, the *hard* end of the path is t near 0
(closest to source), which is the inverse of diffusion's high-t.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowConfig:
    timesteps: int = 1000  # nominal scale used only to drive the existing time embedding
    sigma_noise: float = 0.0  # optional off-path Gaussian noise added to the interpolant
    method: str = "flow"  # serialization tag distinguishing this from "diffusion"
    # Training-time t-sampling. "uniform" = legacy default (t ~ U[0,1]).
    # "logit_normal" = SD3 / EDM-style; t = sigmoid(N(mu, sigma)), peaked at
    # 0.5, biases training toward the hard middle of the path. mu shifts the
    # peak (>0 → toward t=1, <0 → toward t=0); sigma widens it.
    t_sample_mode: str = "uniform"
    t_sample_mu: float = 0.0
    t_sample_sigma: float = 1.0


class RectifiedImageFlow:
    def __init__(self, config: FlowConfig, device: torch.device):
        if config.timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {config.timesteps}")
        if config.sigma_noise < 0:
            raise ValueError(f"sigma_noise must be >= 0, got {config.sigma_noise}")
        self.config = config
        self.device = device

    # ---- t scaling helpers ----------------------------------------------------------------

    def _scale_t(self, t_cont: torch.Tensor) -> torch.Tensor:
        return t_cont.float() * float(self.config.timesteps - 1)

    def _t_range_continuous(self, t_low: int, t_high: int | None) -> tuple[float, float]:
        T = self.config.timesteps
        if t_high is None:
            t_high = T
        t_low = max(0, min(t_low, T - 1))
        t_high = max(t_low + 1, min(t_high, T))
        return t_low / T, t_high / T

    # ---- path / interpolant ---------------------------------------------------------------

    def q_sample(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        t_cont: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build x_t = (1-t)*source + t*target + sigma*noise."""
        t_b = t_cont.view(-1, 1, 1, 1).float()
        x_t = (1.0 - t_b) * source + t_b * target
        if self.config.sigma_noise > 0:
            if noise is None:
                noise = torch.randn_like(target)
            x_t = x_t + self.config.sigma_noise * noise
        elif noise is None:
            noise = torch.zeros_like(target)
        return x_t, noise

    def predict_target_from_v(
        self,
        x_t: torch.Tensor,
        t_cont: torch.Tensor,
        v_hat: torch.Tensor,
    ) -> torch.Tensor:
        """If the velocity stays constant from now to t=1, the predicted target is x_t + (1-t)*v_hat."""
        t_b = t_cont.view(-1, 1, 1, 1).float()
        x1 = x_t + (1.0 - t_b) * v_hat
        return x1.clamp(0.0, 1.0)

    # ---- source dropout (CFG-style) -------------------------------------------------------

    def _apply_source_dropout(self, source: torch.Tensor, dropout: float) -> torch.Tensor:
        if dropout <= 0:
            return source
        keep = (torch.rand(source.shape[0], 1, 1, 1, device=source.device) > dropout).float()
        return source * keep

    # ---- training -------------------------------------------------------------------------

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
        contrastive_source_weight: float = 0.0,
        contrastive_source_margin: float = 0.15,
    ):
        b = target.shape[0]
        t_low_c, t_high_c = self._t_range_continuous(t_low, t_high)
        if self.config.t_sample_mode == "logit_normal":
            # SD3-style: t = sigmoid(N(mu, sigma)). Naturally lives in (0, 1),
            # peaked at sigmoid(mu)=0.5 when mu=0. Clamp to the configured
            # range to honor t_low/t_high constraints.
            u = torch.randn(b, device=target.device) * self.config.t_sample_sigma + self.config.t_sample_mu
            t_cont = torch.sigmoid(u).clamp(t_low_c, t_high_c)
        else:
            t_cont = torch.rand(b, device=target.device) * (t_high_c - t_low_c) + t_low_c

        x_t, noise = self.q_sample(source, target, t_cont)
        source_in = self._apply_source_dropout(source, source_dropout)
        t_emb_in = self._scale_t(t_cont)
        v_hat = model(source_in, x_t, t_emb_in)

        v_target = target - source
        flow_loss = F.mse_loss(v_hat, v_target)
        x_target_hat = self.predict_target_from_v(x_t, t_cont, v_hat)

        lpips_loss = torch.tensor(0.0, device=target.device)
        loss = flow_loss
        if aux_lpips is not None and aux_lpips_weight > 0:
            # Reset before every call: LearnedPerceptualImagePatchSimilarity is a
            # stateful torchmetrics Metric that accumulates sum_scores across calls.
            # Without reset, sum_scores grows a grad_fn chain one node deeper each
            # step (two nodes for exp38 which calls aux_lpips twice), making
            # loss.backward() traverse an ever-longer graph → continuous throughput
            # collapse over a 20k-step run. Reset keeps the chain depth constant at 1.
            aux_lpips.reset()
            lpips_loss = aux_lpips(x_target_hat, target).mean()
            loss = loss + aux_lpips_weight * lpips_loss

            # Contrastive: penalize predictions that stay too close to source.
            # Margin form: only contributes when lpips(out, source) < margin —
            # past the margin the term is zero, so we don't reward arbitrarily
            # noisy outputs that happen to be far from source.
            if contrastive_source_weight > 0:
                aux_lpips.reset()
                lpips_source = aux_lpips(x_target_hat, source).mean()
                contrastive = F.relu(contrastive_source_margin - lpips_source)
                loss = loss + contrastive_source_weight * contrastive

        return loss, t_emb_in, x_t, noise, v_hat, x_target_hat, flow_loss.detach(), lpips_loss.detach()

    # ---- inference ------------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model,
        source: torch.Tensor,
        image_size: int = 128,
        channels: int = 3,
        sample_steps: int | None = None,
        log_every: int | None = None,
        cfg_scale: float = 1.0,
    ):
        """Euler integration from x=source at t=0 to x≈target at t=1.

        `image_size` and `channels` are accepted for interface parity with
        GaussianImageDiffusion.sample but ignored: x is initialised from source.
        """
        del image_size, channels
        if sample_steps is None or sample_steps <= 0:
            sample_steps = 10

        b = source.shape[0]
        # Match the training-time x_t distribution: at t=0 training samples are
        # `source + sigma * noise`, so inference must start from the same.
        # Negligible at sigma_noise=0.05 (legacy), important at larger σ.
        x = source.clone()
        if self.config.sigma_noise > 0:
            x = x + self.config.sigma_noise * torch.randn_like(x)
        ts = torch.linspace(0.0, 1.0, sample_steps + 1, device=self.device)
        frames: list[torch.Tensor] = []

        # CFG: at cfg_scale != 1, also do an unconditioned pass (source=zeros,
        # matching training-time source_dropout) and combine:
        #   v = v_uncond + cfg_scale * (v_cond - v_uncond)
        # cfg_scale=1 → conditioned only (default, no extra cost).
        use_cfg = cfg_scale != 1.0
        zero_source = torch.zeros_like(source) if use_cfg else None

        for i in range(sample_steps):
            t_cur = ts[i].expand(b)
            dt = float(ts[i + 1] - ts[i])
            t_emb_in = self._scale_t(t_cur)
            v_cond = model(source, x, t_emb_in)
            if use_cfg:
                v_uncond = model(zero_source, x, t_emb_in)
                v_hat = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_hat = v_cond
            x = x + dt * v_hat
            if log_every is not None and (i % log_every == 0 or i == 0 or i == sample_steps - 1):
                frames.append(x.detach().clamp(0, 1).cpu())

        return x.clamp(0, 1), frames
