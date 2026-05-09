import numpy as np

from src.metrics import mae, pinaw, picp, rmse, smape


def test_point_metrics_return_expected_values():
    y = np.array([1.0, 2.0, 4.0])
    pred = np.array([1.0, 3.0, 2.0])

    assert mae(y, pred) == 1.0
    assert round(rmse(y, pred), 6) == round(np.sqrt(5 / 3), 6)
    assert smape(y, pred) > 0


def test_interval_metrics_measure_coverage_and_width():
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 1.5, 4.0])
    upper = np.array([2.0, 2.5, 5.0])

    assert picp(y, lower, upper) == 2 / 3
    assert pinaw(y, lower, upper) == 2 / 3
