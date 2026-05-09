from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureColumns:
    history: list[str]
    weather: list[str]
    time: list[str]
    direct: list[str]
    target: str = "AC_POWER"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
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
    out["is_daylight"] = ((out["IRRADIATION"] > 0) | (out["AC_POWER"] > 0)).astype(int)

    out["last_ac_power"] = out["AC_POWER"]
    out["last_dc_power"] = out["DC_POWER"]
    out["last_irradiation"] = out["IRRADIATION"]
    capacity = max(float(out["AC_POWER"].max()), 1.0)
    out["ac_power_norm"] = out["AC_POWER"] / capacity
    return out


def split_feature_columns() -> FeatureColumns:
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
    numeric = df.select_dtypes(include="number")
    if target not in numeric:
        raise ValueError(f"target column {target!r} is not numeric or missing")
    return numeric.corr(method="pearson")[target].sort_values(key=lambda s: s.abs(), ascending=False)
