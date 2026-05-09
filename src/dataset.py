from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FeatureColumns


@dataclass
class TimeSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class WindowArrays:
    history: np.ndarray
    weather: np.ndarray
    time: np.ndarray
    direct: np.ndarray
    target: np.ndarray
    target_times: np.ndarray
    history_end_times: np.ndarray


def build_time_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> TimeSplits:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    ordered = df.sort_values("DATE_TIME").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return TimeSplits(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        val=ordered.iloc[train_end:val_end].reset_index(drop=True),
        test=ordered.iloc[val_end:].reset_index(drop=True),
    )


def make_window_arrays(
    df: pd.DataFrame,
    columns: FeatureColumns,
    lookback: int,
    horizon: int,
) -> WindowArrays:
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    required = set(columns.history + columns.weather + columns.time + columns.direct + [columns.target, "DATE_TIME"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"dataframe is missing columns: {missing}")

    ordered = df.sort_values("DATE_TIME").reset_index(drop=True)
    n_samples = len(ordered) - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("not enough rows to build windows")

    history = []
    weather = []
    time_features = []
    direct = []
    target = []
    target_times = []
    history_end_times = []

    for start in range(n_samples):
        hist_end = start + lookback
        target_end = hist_end + horizon
        history.append(ordered.iloc[start:hist_end][columns.history].to_numpy(dtype=np.float32))
        weather.append(ordered.iloc[hist_end:target_end][columns.weather].to_numpy(dtype=np.float32))
        time_features.append(ordered.iloc[hist_end:target_end][columns.time].to_numpy(dtype=np.float32))
        direct.append(ordered.iloc[hist_end - 1][columns.direct].to_numpy(dtype=np.float32))
        target.append(ordered.iloc[hist_end:target_end][columns.target].to_numpy(dtype=np.float32))
        target_times.append(ordered.iloc[hist_end:target_end]["DATE_TIME"].to_numpy())
        history_end_times.append(ordered.iloc[hist_end - 1]["DATE_TIME"])

    return WindowArrays(
        history=np.asarray(history, dtype=np.float32),
        weather=np.asarray(weather, dtype=np.float32),
        time=np.asarray(time_features, dtype=np.float32),
        direct=np.asarray(direct, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        target_times=np.asarray(target_times),
        history_end_times=np.asarray(history_end_times),
    )


def fit_scalers(train: WindowArrays) -> dict[str, StandardScaler]:
    scalers = {
        "history": StandardScaler(),
        "weather": StandardScaler(),
        "time": StandardScaler(),
        "direct": StandardScaler(),
        "target": StandardScaler(),
    }
    scalers["history"].fit(train.history.reshape(-1, train.history.shape[-1]))
    scalers["weather"].fit(train.weather.reshape(-1, train.weather.shape[-1]))
    scalers["time"].fit(train.time.reshape(-1, train.time.shape[-1]))
    scalers["direct"].fit(train.direct)
    scalers["target"].fit(train.target.reshape(-1, 1))
    return scalers


def transform_windows(windows: WindowArrays, scalers: dict[str, StandardScaler]) -> WindowArrays:
    history = scalers["history"].transform(windows.history.reshape(-1, windows.history.shape[-1])).reshape(windows.history.shape)
    weather = scalers["weather"].transform(windows.weather.reshape(-1, windows.weather.shape[-1])).reshape(windows.weather.shape)
    time_features = scalers["time"].transform(windows.time.reshape(-1, windows.time.shape[-1])).reshape(windows.time.shape)
    direct = scalers["direct"].transform(windows.direct)
    target = scalers["target"].transform(windows.target.reshape(-1, 1)).reshape(windows.target.shape)
    return WindowArrays(
        history=history.astype(np.float32),
        weather=weather.astype(np.float32),
        time=time_features.astype(np.float32),
        direct=direct.astype(np.float32),
        target=target.astype(np.float32),
        target_times=windows.target_times,
        history_end_times=windows.history_end_times,
    )


class PVWindowDataset:
    def __init__(self, arrays: WindowArrays):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PVWindowDataset requires torch. Install PyTorch first.") from exc
        self.torch = torch
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.target)

    def __getitem__(self, idx: int):
        torch = self.torch
        return {
            "history": torch.from_numpy(self.arrays.history[idx]),
            "weather": torch.from_numpy(self.arrays.weather[idx]),
            "time": torch.from_numpy(self.arrays.time[idx]),
            "direct": torch.from_numpy(self.arrays.direct[idx]),
            "target": torch.from_numpy(self.arrays.target[idx]),
        }
