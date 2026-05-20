from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pv import FeatureColumns, load_plant_dataframe, make_window_arrays


def test_load_plant_dataframe_aggregates_generation_and_merges_weather(tmp_path: Path):
    generation_path = tmp_path / "generation.csv"
    weather_path = tmp_path / "weather.csv"
    generation_path.write_text(
        "\n".join(
            [
                "DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD",
                "15-05-2020 00:00,1,a,10,2,0,0",
                "15-05-2020 00:00,1,b,20,3,0,0",
                "15-05-2020 00:15,1,a,30,4,0,0",
                "15-05-2020 00:15,1,b,40,5,0,0",
            ]
        ),
        encoding="utf-8",
    )
    weather_path.write_text(
        "\n".join(
            [
                "DATE_TIME,PLANT_ID,SOURCE_KEY,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION",
                "2020-05-15 00:00:00,1,s,25,30,0.1",
                "2020-05-15 00:15:00,1,s,26,31,0.2",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_plant_dataframe(generation_path, weather_path)

    assert frame["DATE_TIME"].tolist() == pd.to_datetime(["2020-05-15 00:00:00", "2020-05-15 00:15:00"]).tolist()
    assert frame["AC_POWER"].tolist() == [5, 9]
    assert frame["DC_POWER"].tolist() == [30, 70]
    assert frame["IRRADIATION"].tolist() == [0.1, 0.2]


def test_make_window_arrays_uses_history_weather_direct_and_target():
    frame = pd.DataFrame(
        {
            "AC_POWER": [1.0, 2.0, 3.0, 4.0, 5.0],
            "IRRADIATION": [0.1, 0.2, 0.3, 0.4, 0.5],
            "MODULE_TEMPERATURE": [20, 21, 22, 23, 24],
        }
    )
    columns = FeatureColumns(
        history=["AC_POWER"],
        weather=["IRRADIATION", "MODULE_TEMPERATURE"],
        direct=["AC_POWER"],
        target="AC_POWER",
    )

    arrays = make_window_arrays(frame, columns, lookback=2, horizon=2)

    assert arrays.history.shape == (2, 2, 1)
    assert arrays.weather.shape == (2, 2, 2)
    assert arrays.direct.shape == (2, 1)
    assert arrays.target.shape == (2, 2)
    np.testing.assert_array_equal(arrays.history[0, :, 0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(arrays.weather[0, :, 0], np.array([0.3, 0.4], dtype=np.float32))
    np.testing.assert_array_equal(arrays.direct[0], np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(arrays.target[0], np.array([3.0, 4.0], dtype=np.float32))

