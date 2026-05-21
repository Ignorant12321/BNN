"""统一预测输出。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.scaling import inverse_target_prediction, inverse_target_values
from src.torch_runtime import import_torch


def predict_dataframe(label: str, model, arrays, config: dict | None = None) -> pd.DataFrame:
    """把模型预测展开为 DataFrame，便于保存和后续画图。"""
    mean, log_var = predict_arrays(model, arrays, config=config)
    scaler = (config or {}).get("data", {}).get("scaling", {}).get("scaler")
    mean, log_var = inverse_target_prediction(mean, log_var, scaler)
    target = inverse_target_values(arrays.target[: len(mean)], scaler)
    rows = []
    target_time = getattr(arrays, "target_time", None)
    for sample_index in range(len(mean)):
        for horizon_index in range(mean.shape[1]):
            row = {
                "label": label,
                "sample": sample_index,
                "horizon": horizon_index,
                "target": float(target[sample_index, horizon_index]),
                "mean": float(mean[sample_index, horizon_index]),
                "log_var": float(log_var[sample_index, horizon_index]),
            }
            if target_time is not None:
                row["target_time"] = str(target_time[sample_index, horizon_index])
            rows.append(row)
    return pd.DataFrame(rows)


def predict_arrays(model, arrays, config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 NumPy 格式预测，自动适配 NumPy 和 PyTorch 模型。"""
    if getattr(model, "is_torch_model", False):
        torch = import_torch()
        device = next(model.parameters()).device
        n_samples = int((config or {}).get("evaluation", {}).get("n_samples", 1))
        if getattr(model, "stochastic_predict", False):
            n_samples = max(n_samples, 2)
            return predict_torch_samples(model, arrays, device, n_samples)
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


def predict_torch_samples(model, arrays, device, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """多次随机 forward，合并 aleatoric 与 epistemic 不确定性。"""
    torch = import_torch()
    was_training = model.training
    means = []
    variances = []
    model.train()
    with torch.no_grad():
        batch = {
            "history": torch.from_numpy(arrays.history.astype(np.float32)).to(device),
            "weather": torch.from_numpy(arrays.weather.astype(np.float32)).to(device),
            "direct": torch.from_numpy(arrays.direct.astype(np.float32)).to(device),
        }
        for _ in range(int(n_samples)):
            mean, log_var = model(batch)
            means.append(mean.detach().cpu().numpy())
            variances.append(torch.exp(log_var).detach().cpu().numpy())
    if not was_training:
        model.eval()
    mean_stack = np.stack(means, axis=0)
    variance_stack = np.stack(variances, axis=0)
    mean = np.mean(mean_stack, axis=0)
    total_variance = np.mean(variance_stack + mean_stack**2, axis=0) - mean**2
    log_var = np.log(np.maximum(total_variance, 1e-8))
    return mean.astype(np.float32), log_var.astype(np.float32)
