from __future__ import annotations

import numpy as np

from src.metrics import gaussian_nll_np, mae, nrmse, pinaw, picp, rmse, smape
from src.predict import interval_from_samples


def evaluate_predictions(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    lower90, upper90 = interval_from_samples(samples, 0.90)
    lower95, upper95 = interval_from_samples(samples, 0.95)
    return {
        "mae": mae(y_true, mean),
        "rmse": rmse(y_true, mean),
        "nrmse": nrmse(y_true, mean),
        "smape": smape(y_true, mean),
        "nll": gaussian_nll_np(y_true, mean, std**2),
        "picp_90": picp(y_true, lower90, upper90),
        "pinaw_90": pinaw(y_true, lower90, upper90),
        "picp_95": picp(y_true, lower95, upper95),
        "pinaw_95": pinaw(y_true, lower95, upper95),
    }
