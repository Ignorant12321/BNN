"""Train-split fitted scaling for window arrays."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.data.pv import WindowArrays


ScalerPayload = dict[str, dict[str, Any]]


def fit_window_scaler(arrays_by_split: dict[str, WindowArrays]) -> ScalerPayload:
    """Fit standard scalers from the train split only."""
    train = arrays_by_split["train"]
    return {
        "method": {"name": "standard"},
        "history": _fit_feature_scaler(train.history),
        "weather": _fit_feature_scaler(train.weather),
        "direct": _fit_feature_scaler(train.direct),
        "target": _fit_scalar_scaler(train.target),
    }


def transform_window_arrays_by_split(
    arrays_by_split: dict[str, WindowArrays],
    scaler: ScalerPayload,
) -> dict[str, WindowArrays]:
    """Apply a fitted scaler to all split arrays."""
    return {split_name: transform_window_arrays(arrays, scaler) for split_name, arrays in arrays_by_split.items()}


def transform_window_arrays(arrays: WindowArrays, scaler: ScalerPayload) -> WindowArrays:
    """Scale model inputs and target while preserving metadata."""
    return WindowArrays(
        history=_transform_feature_array(arrays.history, scaler["history"]),
        weather=_transform_feature_array(arrays.weather, scaler["weather"]),
        direct=_transform_feature_array(arrays.direct, scaler["direct"]),
        target=_transform_scalar_array(arrays.target, scaler["target"]),
        target_time=arrays.target_time,
    )


def inverse_target_prediction(
    mean: np.ndarray,
    log_var: np.ndarray,
    scaler: ScalerPayload | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert target-space mean and log variance back to raw units."""
    if not scaler:
        return mean, log_var
    target_scaler = scaler["target"]
    std = float(target_scaler["std"])
    restored_mean = inverse_target_values(mean, scaler)
    restored_log_var = log_var + np.float32(2.0 * np.log(std))
    return restored_mean.astype(np.float32), restored_log_var.astype(np.float32)


def inverse_target_values(values: np.ndarray, scaler: ScalerPayload | None) -> np.ndarray:
    """Convert scaled target values back to raw units."""
    if not scaler:
        return values
    target_scaler = scaler["target"]
    return (values * float(target_scaler["std"]) + float(target_scaler["mean"])).astype(np.float32)


def should_scale_torch_training(config: dict[str, Any]) -> bool:
    """Return whether the current training config should standardize arrays."""
    data_scaling = config.get("data", {}).get("scaling", {})
    if isinstance(data_scaling, dict) and "enabled" in data_scaling:
        return bool(data_scaling["enabled"])
    return str(config.get("training", {}).get("backend", "numpy")).lower() == "torch"


def attach_fitted_scaler(config: dict[str, Any], scaler: ScalerPayload) -> dict[str, Any]:
    """Return a config copy containing serialized scaling parameters."""
    result = deepcopy(config)
    result.setdefault("data", {})
    result["data"]["scaling"] = {"enabled": True, "scaler": _to_builtin(scaler)}
    return result


def scaler_from_config(config: dict[str, Any]) -> ScalerPayload | None:
    """Read fitted scaling parameters from a saved config if present."""
    scaling = config.get("data", {}).get("scaling")
    if not isinstance(scaling, dict) or not scaling.get("enabled", False):
        return None
    scaler = scaling.get("scaler")
    if not isinstance(scaler, dict):
        return None
    return scaler


def _fit_feature_scaler(values: np.ndarray) -> dict[str, Any]:
    if values.shape[-1] == 0:
        return {"mean": [], "std": []}
    axes = tuple(range(values.ndim - 1))
    mean = np.mean(values, axis=axes)
    std = _safe_std(np.std(values, axis=axes))
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def _fit_scalar_scaler(values: np.ndarray) -> dict[str, Any]:
    mean = float(np.mean(values))
    std = float(_safe_std(np.asarray(np.std(values), dtype=np.float32)))
    return {"mean": mean, "std": std}


def _transform_feature_array(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _transform_scalar_array(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return ((values - float(scaler["mean"])) / float(scaler["std"])).astype(np.float32)


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.where(std > 0, std, 1.0).astype(np.float32)


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
