from pathlib import Path

import pandas as pd

from src.data.preprocess import run_preprocess
from src.data.split import run_split


def _write_raw_csvs(tmp_path: Path) -> tuple[Path, Path]:
    generation_path = tmp_path / "generation.csv"
    weather_path = tmp_path / "weather.csv"
    generation_rows = ["DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD"]
    weather_rows = ["DATE_TIME,PLANT_ID,SOURCE_KEY,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION"]
    for idx in range(8):
        generation_rows.append(f"15-05-2020 0{idx // 4}:{(idx % 4) * 15:02d},1,a,{idx + 1},{idx + 2},0,0")
        weather_rows.append(f"2020-05-15 0{idx // 4}:{(idx % 4) * 15:02d}:00,1,s,25,30,{idx / 10}")
    generation_path.write_text("\n".join(generation_rows), encoding="utf-8")
    weather_path.write_text("\n".join(weather_rows), encoding="utf-8")
    return generation_path, weather_path


def test_run_preprocess_writes_merged_csv(tmp_path: Path):
    generation_path, weather_path = _write_raw_csvs(tmp_path)
    config = {
        "data": {
            "generation_path": str(generation_path),
            "weather_path": str(weather_path),
            "processed_dir": str(tmp_path / "processed"),
        }
    }

    output_path = run_preprocess(config)

    assert output_path == tmp_path / "processed" / "plant_frame.csv"
    frame = pd.read_csv(output_path)
    assert list(frame.columns) == [
        "DATE_TIME",
        "SOURCE_COUNT",
        "DC_POWER",
        "AC_POWER",
        "DAILY_YIELD",
        "TOTAL_YIELD",
        "EXPECTED_SOURCE_COUNT",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_generation_time",
    ]
    assert len(frame) == 8


def test_run_split_writes_chronological_train_val_test_csvs(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    frame = pd.DataFrame({"DATE_TIME": pd.date_range("2020-01-01", periods=10, freq="15min"), "AC_POWER": range(10)})
    frame.to_csv(processed_dir / "plant_frame.csv", index=False)
    config = {"data": {"processed_dir": str(processed_dir), "train_ratio": 0.6, "val_ratio": 0.2}}

    outputs = run_split(config)

    assert outputs["train"].name == "train.csv"
    assert outputs["val"].name == "val.csv"
    assert outputs["test"].name == "test.csv"
    assert len(pd.read_csv(outputs["train"])) == 6
    assert len(pd.read_csv(outputs["val"])) == 2
    assert len(pd.read_csv(outputs["test"])) == 2
