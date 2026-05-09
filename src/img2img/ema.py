from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        ema_params = dict(self.model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            if name not in ema_params:
                continue
            ema_params[name].lerp_(param.detach(), 1.0 - self.decay)

        ema_buffers = dict(self.model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buf in model_buffers.items():
            if name in ema_buffers:
                ema_buffers[name].copy_(buf)

