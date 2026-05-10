"""时间切分、滑动窗口和 PyTorch Dataset。

光伏预测是时间序列任务，最需要避免的是数据泄漏。因此本模块采取两个原则：

1. 先按时间顺序切分 train/val/test，再在各自子集内部构造窗口。
2. scaler 只在训练集上 fit，再用于验证集和测试集 transform。

窗口形式为：过去 `lookback` 步作为历史输入，未来 `horizon` 步作为预测目标。
当前默认配置是过去 32 个 15 分钟点预测未来 16 个 15 分钟点，即 8h -> 4h。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FeatureColumns


@dataclass
class TimeSplits:
    """按时间顺序切出的训练、验证、测试数据表。"""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class WindowArrays:
    """模型输入窗口数组集合。

    history: [样本数, lookback, 历史特征数]
    weather: [样本数, horizon, 天气特征数]
    time: [样本数, horizon, 时间特征数]
    direct: [样本数, 直接输入特征数]
    target: [样本数, horizon]
    """

    history: np.ndarray
    weather: np.ndarray
    time: np.ndarray
    direct: np.ndarray
    target: np.ndarray
    target_times: np.ndarray
    history_end_times: np.ndarray


def build_time_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> TimeSplits:
    """按时间顺序切分数据集。

    不使用随机切分，因为随机切分会让模型在训练阶段看到测试集附近的时间点，
    对时间序列预测是不合理的数据泄漏。
    """
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
    use_future_weather: bool = False,
) -> WindowArrays:
    """把连续时间序列转换成监督学习样本。

    对于起点 start：
    - history 使用 [start, start + lookback)。
    - target/time 使用 [start + lookback, start + lookback + horizon)。
    - weather 默认使用 history 最后一个时刻并在 horizon 内持久化，避免未来实测气象泄漏。
    - direct 使用 history 的最后一个时刻，即 start + lookback - 1。

    如果已接入真实数值天气预报，可设置 use_future_weather=True 使用目标窗口天气特征。
    """
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
        # 历史序列输入给 CNN 分支，保留二维结构 [time, feature]。
        history.append(ordered.iloc[start:hist_end][columns.history].to_numpy(dtype=np.float32))
        # 默认没有真实 NWP 时，不能使用目标窗口内的实测天气；用最近观测值做持久化基线。
        if use_future_weather:
            weather_window = ordered.iloc[hist_end:target_end][columns.weather].to_numpy(dtype=np.float32)
        else:
            last_weather = ordered.iloc[[hist_end - 1]][columns.weather].to_numpy(dtype=np.float32)
            weather_window = np.repeat(last_weather, repeats=horizon, axis=0)
        weather.append(weather_window)
        # 时间特征由目标时间戳确定，不属于未来观测泄漏。
        time_features.append(ordered.iloc[hist_end:target_end][columns.time].to_numpy(dtype=np.float32))
        # direct 分支只取预测点前一时刻，模拟论文中的“直接输入部分”。
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
    """在训练窗口上拟合所有标准化器。

    每类输入的 shape 不同，因此分别拟合 scaler。序列类输入先展平时间维，
    让 scaler 学到每个特征列的全局均值和标准差。
    """
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
    """使用训练集 scaler 标准化窗口数组。"""
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
    """把 WindowArrays 包装成 PyTorch Dataset。

    这里延迟导入 torch，是为了让没有安装 PyTorch 的环境仍能运行数据处理
    和指标相关测试。
    """

    def __init__(self, arrays: WindowArrays):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PVWindowDataset requires torch. Install PyTorch first.") from exc
        self.arrays = arrays

    def __len__(self) -> int:
        """返回样本窗口数量。"""
        return len(self.arrays.target)

    def __getitem__(self, idx: int):
        """返回一个训练样本，键名与模型 forward 中读取的键名一致。"""
        import torch

        return {
            "history": torch.from_numpy(self.arrays.history[idx]),
            "weather": torch.from_numpy(self.arrays.weather[idx]),
            "time": torch.from_numpy(self.arrays.time[idx]),
            "direct": torch.from_numpy(self.arrays.direct[idx]),
            "target": torch.from_numpy(self.arrays.target[idx]),
        }
