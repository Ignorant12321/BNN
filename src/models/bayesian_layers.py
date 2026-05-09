from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return F.linear(x, weight, bias)

    def kl_loss(self) -> torch.Tensor:
        return self._kl_normal(self.weight_mu, self.weight_rho) + self._kl_normal(self.bias_mu, self.bias_rho)

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def _kl_normal(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        prior_sigma = torch.as_tensor(self.prior_sigma, dtype=mu.dtype, device=mu.device)
        prior_mu = torch.as_tensor(self.prior_mu, dtype=mu.dtype, device=mu.device)
        return torch.sum(
            torch.log(prior_sigma / sigma)
            + (sigma.pow(2) + (mu - prior_mu).pow(2)) / (2 * prior_sigma.pow(2))
            - 0.5
        )
