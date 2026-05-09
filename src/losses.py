from __future__ import annotations

import math

import torch


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_var = torch.clamp(log_var, min=-10.0, max=6.0)
    inv_var = torch.exp(-log_var)
    return 0.5 * torch.mean(log_var + (target - mean) ** 2 * inv_var + math.log(2 * math.pi))


def elbo_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    num_batches: int,
) -> torch.Tensor:
    return gaussian_nll(mean, log_var, target) + beta * kl / max(num_batches, 1)
