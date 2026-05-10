"""特征工程模块。

这里把论文中的输入特征分成四组：

1. history: 预测点前 4 小时 AC_POWER 历史序列。
2. weather: 预测窗口内的天气预报序列和小时数。
3. direct: 预测点前一刻的 AC_POWER。

这些分组会被 `src.dataset.make_window_arrays` 和主模型共同使用，因此集中
维护在本文件中，避免训练和推理阶段使用不一致的列名。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureColumns:
    """模型输入特征列分组。

    frozen=True 表示实例创建后不再修改，防止训练过程中误改列配置。
    """

    history: list[str]
    weather: list[str]
    time: list[str]
    direct: list[str]
    target: str = "AC_POWER"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间特征、白天标记和直接输入特征。

    模型的天气序列包含数值小时 `hour`，用于表示预测窗口内每个点
    所处的日内时刻。sin/cos 周期编码保留给 notebook 或相关性分析。
    """
    out = df.copy()
    dt = pd.to_datetime(out["DATE_TIME"])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    day = dt.dt.dayofyear.astype(float)
    month = dt.dt.month.astype(float)

    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dayofyear_sin"] = np.sin(2 * np.pi * day / 366)
    out["dayofyear_cos"] = np.cos(2 * np.pi * day / 366)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    # 白天标记用于分析或后续扩展评估。只要辐照度或 AC 出力大于 0，
    # 就认为该时刻属于有效发电时段。
    out["is_daylight"] = ((out["IRRADIATION"] > 0) | (out["AC_POWER"] > 0)).astype(int)

    # direct 分支只允许看到预测点前一刻的 AC_POWER。窗口构造时会从
    # 历史窗口最后一行取这一列，避免误取预测点或预测点之后的数据。
    out["last_ac_power"] = out["AC_POWER"]
    return out


def split_feature_columns() -> FeatureColumns:
    """返回当前模型使用的默认特征分组。"""
    return FeatureColumns(
        history=["AC_POWER"],
        weather=[
            "IRRADIATION",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "hour",
        ],
        time=[],
        direct=["last_ac_power"],
    )


def pearson_correlations(df: pd.DataFrame, target: str = "AC_POWER") -> pd.Series:
    """计算所有数值特征与目标列的 Pearson 相关系数。

    该函数主要服务论文中的“输入特征相关性分析”部分，可用于判断哪些
    特征与光伏出力关系更强。
    """
    numeric = df.select_dtypes(include="number")
    if target not in numeric:
        raise ValueError(f"target column {target!r} is not numeric or missing")
    return numeric.corr(method="pearson")[target].sort_values(key=lambda s: s.abs(), ascending=False)
