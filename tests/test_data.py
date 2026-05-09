from pathlib import Path

import pandas as pd

from src.data import load_generation_data, load_weather_data, prepare_plant_dataframe


def test_load_generation_data_parses_day_first_timestamp(tmp_path: Path):
    path = tmp_path / "generation.csv"
    path.write_text(
        "DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD\n"
        "15-05-2020 00:00,1,A,0,0,0,10\n",
        encoding="utf-8",
    )

    df = load_generation_data(path)

    assert df.loc[0, "DATE_TIME"] == pd.Timestamp("2020-05-15 00:00:00")
    assert df.loc[0, "AC_POWER"] == 0


def test_prepare_plant_dataframe_aggregates_inverters_and_merges_weather(tmp_path: Path):
    gen_path = tmp_path / "generation.csv"
    weather_path = tmp_path / "weather.csv"
    gen_path.write_text(
        "DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD\n"
        "15-05-2020 00:00,1,A,10,8,1,10\n"
        "15-05-2020 00:00,1,B,20,18,2,20\n"
        "15-05-2020 00:15,1,A,12,9,3,30\n"
        "15-05-2020 00:15,1,B,21,19,4,40\n",
        encoding="utf-8",
    )
    weather_path.write_text(
        "DATE_TIME,PLANT_ID,SOURCE_KEY,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION\n"
        "2020-05-15 00:00:00,1,W,25,23,0.0\n"
        "2020-05-15 00:15:00,1,W,26,24,0.1\n",
        encoding="utf-8",
    )

    gen = load_generation_data(gen_path)
    weather = load_weather_data(weather_path)
    df = prepare_plant_dataframe(gen, weather)

    assert list(df["AC_POWER"]) == [26, 28]
    assert list(df["INVERTER_COUNT"]) == [2, 2]
    assert "IRRADIATION" in df.columns
    assert df["DATE_TIME"].is_monotonic_increasing
