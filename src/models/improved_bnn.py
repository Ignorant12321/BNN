from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.bayesian_layers import BayesianLinear
from src.models.branches import DirectInputBranch, HistoryCNNBranch, SequenceMLPBranch


class ImprovedBayesianPVNet(nn.Module):
    def __init__(
        self,
        history_features: int,
        weather_features: int,
        time_features: int,
        direct_features: int,
        horizon: int,
        hidden_dim: int = 128,
        branch_dim: int = 64,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.history_branch = HistoryCNNBranch(history_features, hidden_dim=branch_dim, out_dim=branch_dim)
        self.weather_branch = SequenceMLPBranch(weather_features, horizon=horizon, hidden_dim=hidden_dim, out_dim=branch_dim)
        self.time_branch = SequenceMLPBranch(time_features, horizon=horizon, hidden_dim=hidden_dim, out_dim=branch_dim)
        self.direct_branch = DirectInputBranch(direct_features, hidden_dim=branch_dim, out_dim=branch_dim // 2)
        fusion_dim = branch_dim * 3 + branch_dim // 2
        self.bayes1 = BayesianLinear(fusion_dim, hidden_dim, prior_sigma=prior_sigma)
        self.bayes2 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma=prior_sigma)
        self.mean_head = BayesianLinear(hidden_dim, horizon, prior_sigma=prior_sigma)
        self.log_var_head = BayesianLinear(hidden_dim, horizon, prior_sigma=prior_sigma)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat(
            [
                self.history_branch(batch["history"]),
                self.weather_branch(batch["weather"]),
                self.time_branch(batch["time"]),
                self.direct_branch(batch["direct"]),
            ],
            dim=-1,
        )
        z = F.relu(self.bayes1(z))
        z = F.relu(self.bayes2(z))
        mean = self.mean_head(z)
        log_var = torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)
        return mean, log_var

    def kl_loss(self) -> torch.Tensor:
        kl = torch.zeros((), device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_loss()
        return kl
