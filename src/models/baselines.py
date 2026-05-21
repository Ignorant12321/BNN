"""NumPy 概率预测基线模型。

功能：
    提供无需 PyTorch 的可训练 ridge regression 基线，统一暴露：
    `model.fit(arrays)`、`mean, log_var = model(batch)`、`model.kl_loss()`。

使用：
    通过模型注册表构造：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml
"""

from __future__ import annotations

import numpy as np


class RidgeProbabilisticModel:
    """多输出 ridge regression，接口与当前 PyTorch 概率模型保持一致。"""

    def __init__(self, horizon: int, alpha: float = 1e-3, log_var_floor: float = -10.0):
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.log_var_floor = float(log_var_floor)
        self.coef_: np.ndarray | None = None
        self.log_var_: np.ndarray | None = None

    def __call__(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        features = self.features_from_batch(batch)
        if self.coef_ is None:
            mean = np.zeros((len(features), self.horizon), dtype=np.float32)
        else:
            mean = _add_intercept(features) @ self.coef_
            mean = mean.astype(np.float32)
        log_var_values = self.log_var_ if self.log_var_ is not None else np.zeros(self.horizon, dtype=np.float32)
        log_var = np.broadcast_to(log_var_values, mean.shape).astype(np.float32)
        return mean, log_var

    def fit(self, arrays) -> None:
        """根据窗口数组拟合回归权重和残差方差。"""
        batch = {"history": arrays.history, "weather": arrays.weather, "direct": arrays.direct}
        x = _add_intercept(self.features_from_batch(batch))
        y = arrays.target.astype(np.float32)
        penalty = self.alpha * np.eye(x.shape[1], dtype=np.float32)
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
        residual = y - x @ self.coef_
        variance = np.maximum(np.mean(residual**2, axis=0), np.exp(self.log_var_floor))
        self.log_var_ = np.log(variance).astype(np.float32)

    def features_from_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def kl_loss(self) -> float:
        return 0.0


class FlattenedMLPBaseline(RidgeProbabilisticModel):
    """使用全部展平输入的线性基线，对应 MLP 模型族的轻量版本。"""

    def features_from_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                batch["history"].reshape(len(batch["history"]), -1),
                batch["weather"].reshape(len(batch["weather"]), -1),
                batch["direct"].reshape(len(batch["direct"]), -1),
            ],
            axis=1,
        ).astype(np.float32)


class HistoryCNNBaseline(RidgeProbabilisticModel):
    """只使用历史序列的线性基线，对应 CNN 模型族的轻量版本。"""

    def features_from_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        return batch["history"].reshape(len(batch["history"]), -1).astype(np.float32)


class MCDropoutBaseline(RidgeProbabilisticModel):
    """使用历史和天气输入的基线，对应 MC Dropout 模型族的轻量版本。"""

    def features_from_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                batch["history"].reshape(len(batch["history"]), -1),
                batch["weather"].reshape(len(batch["weather"]), -1),
            ],
            axis=1,
        ).astype(np.float32)


def _add_intercept(features: np.ndarray) -> np.ndarray:
    intercept = np.ones((len(features), 1), dtype=np.float32)
    return np.concatenate([intercept, features.astype(np.float32)], axis=1)
