"""评价指标测试。"""

import numpy as np

from src.metrics import gaussian_crps_np, mae, pinaw, picp, rmse, smape


def test_point_metrics_return_expected_values():
    """MAE、RMSE、sMAPE 应返回稳定的数值。"""
    y = np.array([1.0, 2.0, 4.0])
    pred = np.array([1.0, 3.0, 2.0])

    assert mae(y, pred) == 1.0
    assert round(rmse(y, pred), 6) == round(np.sqrt(5 / 3), 6)
    assert smape(y, pred) > 0


def test_interval_metrics_measure_coverage_and_width():
    """PICP 衡量覆盖率，PINAW 衡量归一化区间宽度。"""
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 1.5, 4.0])
    upper = np.array([2.0, 2.5, 5.0])

    assert picp(y, lower, upper) == 2 / 3
    assert pinaw(y, lower, upper) == 2 / 3


def test_gaussian_crps_rewards_sharp_calibrated_predictions():
    """Gaussian CRPS should be small for an exact mean with finite uncertainty."""
    y = np.array([0.0])
    mean = np.array([0.0])
    std = np.array([1.0])

    expected = (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi)

    assert round(gaussian_crps_np(y, mean, std), 6) == round(expected, 6)
