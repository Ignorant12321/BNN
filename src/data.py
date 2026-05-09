"""数据读取与电站级数据表构造。

本模块负责把原始 CSV 转换成后续特征工程和滑动窗口可以直接使用的
时间序列 DataFrame。原始发电数据是一台电站下多个逆变器的记录，
因此这里会先按 `DATE_TIME` 聚合为电站级出力，再与气象传感器数据
按时间戳合并。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# 发电数据必须包含的原始字段。提前校验字段可以避免后续聚合时报出
# 难以定位的 KeyError。
GENERATION_COLUMNS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
]

# 气象数据必须包含的原始字段。
WEATHER_COLUMNS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]


def load_generation_data(path: str | Path) -> pd.DataFrame:
    """读取发电 CSV，并解析日-月-年格式的时间戳。

    Plant_1_Generation_Data.csv 的 `DATE_TIME` 形如 `15-05-2020 00:00`，
    如果直接让 pandas 自动推断，可能在不同地区设置下被误解为月-日-年。
    因此这里显式指定 `%d-%m-%Y %H:%M`。
    """
    df = pd.read_csv(path)
    _require_columns(df, GENERATION_COLUMNS, "generation")
    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    return df.sort_values(["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)


def load_weather_data(path: str | Path) -> pd.DataFrame:
    """读取气象 CSV，并解析标准时间戳。

    气象文件中的时间戳形如 `2020-05-15 00:00:00`，pandas 可以稳定解析。
    排序后返回，保证后续 merge 和插值逻辑都在时间有序的前提下运行。
    """
    df = pd.read_csv(path)
    _require_columns(df, WEATHER_COLUMNS, "weather")
    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df.sort_values("DATE_TIME").reset_index(drop=True)


def prepare_plant_dataframe(
    generation: pd.DataFrame,
    weather: pd.DataFrame,
    fill_missing: bool = True,
    freq: str = "15min",
) -> pd.DataFrame:
    """构造电站级建模数据表。

    参数:
        generation: 已解析时间戳的发电数据，包含多个逆变器记录。
        weather: 已解析时间戳的气象传感器数据。
        fill_missing: 是否补齐 15 分钟规则时间轴并对数值列插值。
        freq: 时间轴频率，当前数据集为 15 分钟。

    返回:
        每个时间戳一行的电站级数据表，包含聚合后的功率、累计发电量、
        逆变器数量和气象变量。
    """
    # 发电文件中同一时刻有多个 SOURCE_KEY，即多个逆变器。预测目标是电站
    # 总出力，所以 AC/DC 功率按时间求和。
    plant = (
        generation.groupby("DATE_TIME", as_index=False)
        .agg(
            DC_POWER=("DC_POWER", "sum"),
            AC_POWER=("AC_POWER", "sum"),
            DAILY_YIELD=("DAILY_YIELD", "sum"),
            TOTAL_YIELD=("TOTAL_YIELD", "sum"),
            INVERTER_COUNT=("SOURCE_KEY", "nunique"),
        )
        .sort_values("DATE_TIME")
    )
    weather_cols = [
        "DATE_TIME",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
    ]
    # 使用 inner join 只保留发电与气象都存在的时刻，避免模型输入缺少天气。
    merged = plant.merge(weather[weather_cols], on="DATE_TIME", how="inner")
    merged = merged.sort_values("DATE_TIME").reset_index(drop=True)
    if fill_missing and not merged.empty:
        merged = _fill_regular_time_grid(merged, freq=freq)
    return merged


def load_plant_dataframe(
    generation_path: str | Path = "dataset/Plant_1_Generation_Data.csv",
    weather_path: str | Path = "dataset/Plant_1_Weather_Sensor_Data.csv",
    fill_missing: bool = True,
) -> pd.DataFrame:
    """从默认文件路径直接读取并返回电站级数据表。

    训练入口一般调用这个函数；测试或 notebook 中也可以传入其他 CSV 路径。
    """
    generation = load_generation_data(generation_path)
    weather = load_weather_data(weather_path)
    return prepare_plant_dataframe(generation, weather, fill_missing=fill_missing)


def _fill_regular_time_grid(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """补齐规则时间轴，并对数值列做时间插值。

    原始数据中存在少量不连续的 15 分钟点。滑动窗口模型要求时间步长一致，
    因此这里先 reindex 到完整时间轴，再对数值列进行 time interpolation。
    前后边界的缺失值使用前向/后向填充兜底。
    """
    full_index = pd.date_range(df["DATE_TIME"].min(), df["DATE_TIME"].max(), freq=freq)
    out = df.set_index("DATE_TIME").reindex(full_index)
    out.index.name = "DATE_TIME"
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].interpolate(method="time").ffill().bfill()
    return out.reset_index()


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """校验输入文件字段是否完整。"""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} data is missing columns: {missing}")
