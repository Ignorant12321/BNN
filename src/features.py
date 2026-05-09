"""特征工程模块。

这里把论文中的输入特征分成四组：

1. history: 历史序列特征，交给 1D-CNN 分支处理。
2. weather: 预测窗口内的气象变量，交给 MLP 分支处理。
3. time: 周期时间特征，描述日内/季节性规律。
4. direct: 预测点前一时刻的强相关变量，直接输入后端融合层。

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
    """添加时间周期、白天标记和直接输入特征。

    时间变量具有周期性。例如 23:45 与 00:00 在实际物理意义上很近，
    如果直接使用小时数 23.75 和 0，会让模型误以为二者距离很远。
    因此使用 sin/cos 编码日内、年内和月内周期。
    """
    out = df.copy()
    dt = pd.to_datetime(out["DATE_TIME"])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    day = dt.dt.dayofyear.astype(float)
    month = dt.dt.month.astype(float)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dayofyear_sin"] = np.sin(2 * np.pi * day / 366)
    out["dayofyear_cos"] = np.cos(2 * np.pi * day / 366)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    # 白天标记用于分析或后续扩展评估。只要辐照度或 AC 出力大于 0，
    # 就认为该时刻属于有效发电时段。
    out["is_daylight"] = ((out["IRRADIATION"] > 0) | (out["AC_POWER"] > 0)).astype(int)

    # 论文提到“预测点前一时刻的数据与预测点相关性最强”。这里把最近
    # 一个历史点的关键变量单独复制成 direct 特征，窗口构造时会取历史
    # 窗口最后一行作为直接输入。
    out["last_ac_power"] = out["AC_POWER"]
    out["last_dc_power"] = out["DC_POWER"]
    out["last_irradiation"] = out["IRRADIATION"]
    capacity = max(float(out["AC_POWER"].max()), 1.0)
    out["ac_power_norm"] = out["AC_POWER"] / capacity
    return out


def split_feature_columns() -> FeatureColumns:
    """返回当前模型使用的默认特征分组。"""
    return FeatureColumns(
        history=[
            "AC_POWER",
            "DC_POWER",
            "IRRADIATION",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
        ],
        weather=[
            "IRRADIATION",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
        ],
        time=[
            "hour_sin",
            "hour_cos",
            "dayofyear_sin",
            "dayofyear_cos",
            "month_sin",
            "month_cos",
        ],
        direct=[
            "last_ac_power",
            "last_dc_power",
            "last_irradiation",
        ],
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
