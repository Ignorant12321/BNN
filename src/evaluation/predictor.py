"""统一预测输出。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.torch_runtime import import_torch


def predict_dataframe(label: str, model, arrays) -> pd.DataFrame:
    """把模型预测展开为 DataFrame，便于保存和后续画图。"""
    mean, log_var = predict_arrays(model, arrays)
    target = arrays.target[: len(mean)]
    rows = []
    for sample_index in range(len(mean)):
        for horizon_index in range(mean.shape[1]):
            rows.append(
                {
                    "label": label,
                    "sample": sample_index,
                    "horizon": horizon_index,
                    "target": float(target[sample_index, horizon_index]),
                    "mean": float(mean[sample_index, horizon_index]),
                    "log_var": float(log_var[sample_index, horizon_index]),
                }
            )
    return pd.DataFrame(rows)


def predict_arrays(model, arrays) -> tuple[np.ndarray, np.ndarray]:
    """返回 NumPy 格式预测，自动适配 NumPy 和 PyTorch 模型。"""
    if getattr(model, "is_torch_model", False):
        torch = import_torch()
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            batch = {
                "history": torch.from_numpy(arrays.history.astype(np.float32)).to(device),
                "weather": torch.from_numpy(arrays.weather.astype(np.float32)).to(device),
                "direct": torch.from_numpy(arrays.direct.astype(np.float32)).to(device),
            }
            mean, log_var = model(batch)
        return mean.detach().cpu().numpy(), log_var.detach().cpu().numpy()
    return model(arrays.as_batch())
