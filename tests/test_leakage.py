import numpy as np
import pandas as pd

from src.dataset import make_window_arrays
from src.features import add_basic_features, split_feature_columns


def test_window_timestamps_do_not_put_future_values_in_history():
    n = 24
    df = pd.DataFrame(
        {
            "DATE_TIME": pd.date_range("2020-05-15", periods=n, freq="15min"),
            "AC_POWER": np.arange(n, dtype=float),
            "DC_POWER": np.arange(n, dtype=float),
            "AMBIENT_TEMPERATURE": np.ones(n),
            "MODULE_TEMPERATURE": np.ones(n),
            "IRRADIATION": np.ones(n),
        }
    )
    arrays = make_window_arrays(add_basic_features(df), split_feature_columns(), lookback=6, horizon=4)

    assert arrays.history_end_times[0] < arrays.target_times[0][0]
    assert arrays.history[0, -1, 0] == 5.0
    assert arrays.target[0, 0] == 6.0
