from __future__ import annotations

from pathlib import Path

import pandas as pd


GENERATION_COLUMNS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
]

WEATHER_COLUMNS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]


def load_generation_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, GENERATION_COLUMNS, "generation")
    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    return df.sort_values(["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)


def load_weather_data(path: str | Path) -> pd.DataFrame:
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
    generation = load_generation_data(generation_path)
    weather = load_weather_data(weather_path)
    return prepare_plant_dataframe(generation, weather, fill_missing=fill_missing)


def _fill_regular_time_grid(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    full_index = pd.date_range(df["DATE_TIME"].min(), df["DATE_TIME"].max(), freq=freq)
    out = df.set_index("DATE_TIME").reindex(full_index)
    out.index.name = "DATE_TIME"
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].interpolate(method="time").ffill().bfill()
    return out.reset_index()


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} data is missing columns: {missing}")
