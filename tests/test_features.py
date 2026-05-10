"""特征工程测试。"""

import pandas as pd

from src.features import add_basic_features, split_feature_columns


def test_add_basic_features_adds_cyclical_time_and_daylight_flags():
    """基础特征应包含数值小时、周期时间编码和白天标记。"""
    df = pd.DataFrame(
        {
            "DATE_TIME": pd.to_datetime(["2020-05-15 00:00", "2020-05-15 12:00"]),
            "AC_POWER": [0.0, 100.0],
            "DC_POWER": [0.0, 110.0],
            "AMBIENT_TEMPERATURE": [20.0, 28.0],
            "MODULE_TEMPERATURE": [18.0, 35.0],
            "IRRADIATION": [0.0, 0.8],
        }
    )

    out = add_basic_features(df)

    assert {"hour", "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"}.issubset(out.columns)
    assert list(out["hour"]) == [0.0, 12.0]
    assert list(out["is_daylight"]) == [0, 1]


def test_split_feature_columns_matches_model_inputs():
    """特征分组应匹配论文图中的三路输入语义。"""
    columns = split_feature_columns()

    assert columns.history == ["AC_POWER"]
    assert columns.weather == [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "hour",
    ]
    assert columns.time == []
    assert columns.direct == ["last_ac_power"]
