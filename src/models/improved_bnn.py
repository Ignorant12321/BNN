"""改进 BNN 主模型接口。

功能：
    提供 NumPy ridge 版本的三路输入融合概率模型。若配置使用 PyTorch
    后端，同名模型会由 `src.models.torch_models.TorchPVNet` 构造。

使用：
    python -m src.experiments.train --config configs/models/bnn_24h.yaml
"""

from __future__ import annotations

import numpy as np

from src.models.baselines import RidgeProbabilisticModel


class ImprovedBayesianPVNet(RidgeProbabilisticModel):
    """使用 history、future weather 和 direct 三路输入的概率预测模型。"""

    def features_from_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """拼接历史功率、未来天气和上一时刻功率作为模型特征。"""
        return np.concatenate(
            [
                batch["history"].reshape(len(batch["history"]), -1),
                batch["weather"].reshape(len(batch["weather"]), -1),
                batch["direct"].reshape(len(batch["direct"]), -1),
            ],
            axis=1,
        ).astype(np.float32)
