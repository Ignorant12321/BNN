"""Recursive ultra-short-term Bayesian PV forecasting network.

The model predicts one 15-minute point at a time. Each predicted point is
rolled back into the power history before predicting the next point, so a
4-hour forecast with ``horizon=16`` is produced by 16 recursive single-step
forecasts.
"""

from __future__ import annotations

from src.models.torch_models import BayesianConv1d, BayesianLinear, ProbabilisticTorchModel, torch

from torch import nn
from torch.nn import functional as F


class PVRecursiveUltraShortTermIBNN(ProbabilisticTorchModel):
    """Bayesian PV model with rolling-history recursive decoding."""

    stochastic_predict = True
    REQUIRED_WEATHER_FEATURES = (
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_generation_time",
    )

    def __init__(
        self,
        lookback: int,
        horizon: int,
        history_features: int,
        weather_feature_names: list[str],
        direct_features: int,
        hidden_dim: int = 32,
        context_dim: int = 32,
        teacher_forcing_ratio: float = 0.0,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.history_features = int(history_features)
        self.direct_features = int(direct_features)
        self.weather_feature_names = tuple(weather_feature_names)
        self.teacher_forcing_ratio = float(teacher_forcing_ratio)

        if self.lookback <= 0:
            raise ValueError("pv_usibnn_recursive requires a positive lookback")
        if self.horizon <= 0:
            raise ValueError("pv_usibnn_recursive requires a positive horizon")
        if self.history_features != 1:
            raise ValueError("pv_usibnn_recursive currently requires one history feature: AC_POWER")
        if self.direct_features <= 0:
            raise ValueError("pv_usibnn_recursive requires direct AC_POWER input")
        if not 0.0 <= self.teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0.0 and 1.0")

        self.weather_indices = self._feature_indices(self.REQUIRED_WEATHER_FEATURES)
        step_weather_dim = len(self.weather_indices)
        hidden_dim = int(hidden_dim)
        context_dim = int(context_dim)

        self.history_conv1 = BayesianConv1d(1, context_dim, kernel_size=3, padding=1)
        self.history_conv2 = BayesianConv1d(context_dim, context_dim, kernel_size=3, padding=1)
        self.history_pool = nn.AdaptiveAvgPool1d(4)
        self.history_projection = BayesianLinear(context_dim * 4, context_dim)

        self.weather_branch = nn.Sequential(
            BayesianLinear(step_weather_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, context_dim),
            nn.ReLU(),
        )
        self.step_fusion = nn.Sequential(
            BayesianLinear(context_dim + context_dim + 1, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = BayesianLinear(hidden_dim, 1)
        self.log_var_head = BayesianLinear(hidden_dim, 1)

    def forward(self, batch: dict) -> tuple:
        rolling_history = batch["history"]
        weather = batch["weather"]
        target = batch.get("target")
        batch_size = len(rolling_history)

        means = []
        log_vars = []
        previous_power = batch["direct"][:, :1]

        for horizon_index in range(self.horizon):
            history_features = self._encode_history(rolling_history)
            step_weather = weather[:, horizon_index, self.weather_indices]
            step_features = torch.cat(
                [
                    history_features,
                    self.weather_branch(step_weather),
                    previous_power,
                ],
                dim=1,
            )
            z = self.step_fusion(step_features)
            step_mean = self.mean_head(z)
            step_log_var = torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)
            means.append(step_mean)
            log_vars.append(step_log_var)

            next_power = self._next_history_power(step_mean, target, horizon_index, batch_size)
            rolling_history = torch.cat([rolling_history[:, 1:, :], next_power.unsqueeze(-1)], dim=1)
            previous_power = next_power

        return torch.cat(means, dim=1), torch.cat(log_vars, dim=1)

    def _encode_history(self, rolling_history):
        features = rolling_history.transpose(1, 2)
        features = F.relu(self.history_conv1(features))
        features = F.relu(self.history_conv2(features))
        features = self.history_pool(features).reshape(len(rolling_history), -1)
        return F.relu(self.history_projection(features))

    def _next_history_power(self, step_mean, target, horizon_index: int, batch_size: int):
        if not self.training or target is None or self.teacher_forcing_ratio <= 0.0:
            return step_mean
        teacher_value = target[:, horizon_index : horizon_index + 1]
        if self.teacher_forcing_ratio >= 1.0:
            return teacher_value
        mask = torch.rand(batch_size, 1, device=step_mean.device) < self.teacher_forcing_ratio
        return torch.where(mask, teacher_value, step_mean)

    def _feature_indices(self, required_names: tuple[str, ...]) -> tuple[int, ...]:
        missing = [name for name in required_names if name not in self.weather_feature_names]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"pv_usibnn_recursive missing required future feature(s): {missing_text}")
        return tuple(self.weather_feature_names.index(name) for name in required_names)
