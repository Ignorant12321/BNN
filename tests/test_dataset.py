import numpy as np
import pandas as pd

from src.dataset import build_time_splits, make_window_arrays
from src.features import add_basic_features, split_feature_columns


def _frame(n=40):
    ts = pd.date_range("2020-05-15", periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "DATE_TIME": ts,
            "AC_POWER": np.arange(n, dtype=float),
            "DC_POWER": np.arange(n, dtype=float) + 1,
            "AMBIENT_TEMPERATURE": np.linspace(20, 30, n),
            "MODULE_TEMPERATURE": np.linspace(22, 35, n),
            "IRRADIATION": np.linspace(0, 1, n),
        }
    )
    return add_basic_features(df)


def test_make_window_arrays_uses_past_history_and_future_targets():
    df = _frame(30)
    columns = split_feature_columns()

    arrays = make_window_arrays(df, columns, lookback=4, horizon=3)

    assert arrays.history.shape == (24, 4, len(columns.history))
    assert arrays.weather.shape == (24, 3, len(columns.weather))
    assert arrays.time.shape == (24, 3, len(columns.time))
    assert arrays.direct.shape == (24, len(columns.direct))
    assert arrays.target.shape == (24, 3)
    np.testing.assert_array_equal(arrays.target[0], np.array([4.0, 5.0, 6.0]))


def test_build_time_splits_are_chronological():
    df = _frame(100)

    splits = build_time_splits(df, train_ratio=0.6, val_ratio=0.2)

    assert splits.train["DATE_TIME"].max() < splits.val["DATE_TIME"].min()
    assert splits.val["DATE_TIME"].max() < splits.test["DATE_TIME"].min()
