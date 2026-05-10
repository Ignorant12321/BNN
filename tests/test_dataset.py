"""滑动窗口与时间切分测试。"""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.dataset import PVWindowDataset, build_time_splits, make_window_arrays
from src.features import add_basic_features, split_feature_columns


def _importorskip_torch_loadable() -> None:
    try:
        __import__("torch")
    except ModuleNotFoundError:
        pytest.skip("torch is not installed")
    except OSError as exc:
        pytest.skip(f"torch could not be loaded: {exc}")


def _frame(n=40):
    """构造一段规则 15 分钟光伏样例数据。"""
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
    """窗口构造应使用过去 lookback 步预测未来 horizon 步。"""
    df = _frame(30)
    columns = split_feature_columns()

    arrays = make_window_arrays(df, columns, lookback=4, horizon=3)

    assert arrays.history.shape == (24, 4, len(columns.history))
    assert arrays.weather.shape == (24, 3, len(columns.weather))
    assert arrays.time.shape == (24, 3, len(columns.time))
    assert arrays.direct.shape == (24, len(columns.direct))
    assert arrays.target.shape == (24, 3)
    np.testing.assert_array_equal(arrays.target[0], np.array([4.0, 5.0, 6.0]))


def test_make_window_arrays_uses_weather_persistence_by_default():
    """默认不能把目标窗口内的未来实测气象作为模型输入。"""
    df = _frame(12)
    columns = split_feature_columns()

    arrays = make_window_arrays(df, columns, lookback=4, horizon=3)

    expected = np.repeat(
        df.iloc[[3]][columns.weather].to_numpy(dtype=np.float32),
        repeats=3,
        axis=0,
    )
    np.testing.assert_array_equal(arrays.weather[0], expected)


def test_make_window_arrays_can_use_future_weather_when_nwp_available():
    """若接入真实 NWP，可显式使用预测窗口天气特征。"""
    df = _frame(12)
    columns = split_feature_columns()

    arrays = make_window_arrays(df, columns, lookback=4, horizon=3, use_future_weather=True)

    expected = df.iloc[4:7][columns.weather].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(arrays.weather[0], expected)


def test_build_time_splits_are_chronological():
    """训练、验证、测试集必须保持严格时间顺序。"""
    df = _frame(100)

    splits = build_time_splits(df, train_ratio=0.6, val_ratio=0.2)

    assert splits.train["DATE_TIME"].max() < splits.val["DATE_TIME"].min()
    assert splits.val["DATE_TIME"].max() < splits.test["DATE_TIME"].min()


def test_pv_window_dataset_is_picklable_for_multiprocess_workers():
    """Windows DataLoader workers need to pickle the dataset before spawning."""
    _importorskip_torch_loadable()
    df = _frame(12)
    columns = split_feature_columns()
    arrays = make_window_arrays(df, columns, lookback=4, horizon=3)

    pickle.dumps(PVWindowDataset(arrays))
