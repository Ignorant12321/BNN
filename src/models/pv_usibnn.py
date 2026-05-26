"""Ultra-short-term PV improved Bayesian neural network.

The model follows the paper-style three-part IBNN structure while using only
features available in the Plant_1 dataset: historical power, future solar/time
sequence, future temperature sequence, and current power.
"""

from __future__ import annotations

from src.models.torch_models import BayesianConv1d, BayesianLinear, ProbabilisticTorchModel, torch

from torch import nn
from torch.nn import functional as F


class PVUltraShortTermIBNN(ProbabilisticTorchModel):
    """Dataset-aware ultra-short-term Bayesian PV forecasting network."""

    stochastic_predict = True
    SOLAR_TIME_FEATURES = (
        "IRRADIATION",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_generation_time",
    )
    WEATHER_FEATURES = ("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE")

    def __init__(
        self,
        lookback: int,
        horizon: int,
        history_features: int,
        weather_feature_names: list[str],
        direct_features: int,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.weather_feature_names = tuple(weather_feature_names)
        if self.lookback <= 0 or int(history_features) <= 0:
            raise ValueError("pv_usibnn requires a positive lookback history_power input")
        if int(direct_features) <= 0:
            raise ValueError("pv_usibnn requires direct_power input features")

        self.solar_time_indices = self._feature_indices(self.SOLAR_TIME_FEATURES)
        self.weather_indices = self._feature_indices(self.WEATHER_FEATURES)

        solar_time_dim = self.horizon * len(self.solar_time_indices)
        weather_dim = self.horizon * len(self.weather_indices)

        self.solar_time_branch = nn.Sequential(
            BayesianLinear(solar_time_dim, 32),
            nn.ReLU(),
            BayesianLinear(32, 64),
            nn.ReLU(),
            BayesianLinear(64, 16),
            nn.ReLU(),
        )
        self.history_conv1 = BayesianConv1d(history_features, 32, kernel_size=3, padding=1)
        self.history_conv2 = BayesianConv1d(32, 32, kernel_size=3, padding=1)
        self.history_pool = nn.AdaptiveAvgPool1d(4)
        self.history_projection = BayesianLinear(32 * 4, 32)
        self.weather_branch = nn.Sequential(
            BayesianLinear(weather_dim, 32),
            nn.ReLU(),
            BayesianLinear(32, 16),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            BayesianLinear(16 + 32 + 16 + int(direct_features), 32),
            nn.ReLU(),
            BayesianLinear(32, 16),
            nn.ReLU(),
        )
        self.mean_head = BayesianLinear(16, self.horizon)
        self.log_var_head = BayesianLinear(16, self.horizon)

    def forward(self, batch: dict) -> tuple:
        history = batch["history"]
        weather = batch["weather"]
        direct = batch["direct"]

        solar_time = weather[:, :, self.solar_time_indices].reshape(len(history), -1)
        weather_features = weather[:, :, self.weather_indices].reshape(len(history), -1)
        history_features = history.transpose(1, 2)
        history_features = F.relu(self.history_conv1(history_features))
        history_features = F.relu(self.history_conv2(history_features))
        history_features = self.history_pool(history_features).reshape(len(history), -1)
        history_features = F.relu(self.history_projection(history_features))

        features = torch.cat(
            [
                self.solar_time_branch(solar_time),
                history_features,
                self.weather_branch(weather_features),
                direct.reshape(len(history), -1),
            ],
            dim=1,
        )
        z = self.fusion(features)
        return self.mean_head(z), torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)

    def _feature_indices(self, required_names: tuple[str, ...]) -> tuple[int, ...]:
        missing = [name for name in required_names if name not in self.weather_feature_names]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"pv_usibnn missing required future feature(s): {missing_text}")
        return tuple(self.weather_feature_names.index(name) for name in required_names)
