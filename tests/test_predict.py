"""预测结果展示时间段选择测试。"""

import numpy as np

from src.predict import interval_from_mean_std, select_prediction_plot_data


def test_select_prediction_plot_data_skips_night_by_default():
    """默认画图应从第一个有明显出力的时刻开始，而不是夜间零功率。"""
    times = np.array(
        [
            ["2020-06-13T00:00:00", "2020-06-13T00:15:00", "2020-06-13T06:00:00"],
            ["2020-06-13T06:15:00", "2020-06-13T06:30:00", "2020-06-13T06:45:00"],
        ],
        dtype="datetime64[ns]",
    )
    y_true = np.array([[0.0, 0.0, 10.0], [20.0, 30.0, 40.0]])
    mean = y_true + 1.0
    lower = y_true - 1.0
    upper = y_true + 2.0

    view = select_prediction_plot_data(times, y_true, mean, lower, upper, max_points=3)

    assert str(view["times"][0]) == "2020-06-13T06:00:00.000000000"
    np.testing.assert_array_equal(view["y_true"], np.array([10.0, 20.0, 30.0]))
    assert view["reason"] == "auto_daylight"


def test_select_prediction_plot_data_respects_explicit_time_range():
    """显式时间段应优先于自动白天选择。"""
    times = np.array(
        [["2020-06-13T00:00:00", "2020-06-13T00:15:00", "2020-06-13T06:00:00"]],
        dtype="datetime64[ns]",
    )
    y_true = np.array([[0.0, 0.0, 10.0]])
    mean = y_true
    lower = y_true
    upper = y_true

    view = select_prediction_plot_data(
        times,
        y_true,
        mean,
        lower,
        upper,
        start_time="2020-06-13 00:00:00",
        end_time="2020-06-13 00:15:00",
    )

    np.testing.assert_array_equal(view["y_true"], np.array([0.0, 0.0]))
    assert view["reason"] == "configured_time_range"


def test_select_prediction_plot_data_falls_back_when_no_daylight_exists():
    """如果目标段全为夜间零功率，应回退到前 max_points 个点。"""
    times = np.array([["2020-06-13T00:00:00", "2020-06-13T00:15:00"]], dtype="datetime64[ns]")
    y_true = np.zeros((1, 2))

    view = select_prediction_plot_data(times, y_true, y_true, y_true, y_true, max_points=1)

    np.testing.assert_array_equal(view["y_true"], np.array([0.0]))
    assert view["reason"] == "fallback_first_points"


def test_interval_from_mean_std_uses_aleatoric_uncertainty():
    """预测区间应随模型输出标准差变宽，而不是只依赖 MC 均值样本。"""
    mean = np.array([[10.0, 20.0]])
    std = np.array([[2.0, 4.0]])

    lower, upper = interval_from_mean_std(mean, std, 0.95)

    assert lower[0, 0] < 10.0
    assert upper[0, 0] > 10.0
    assert (upper[0, 1] - lower[0, 1]) > (upper[0, 0] - lower[0, 0])
