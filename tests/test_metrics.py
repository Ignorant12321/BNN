import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import generation_period_metrics, regression_metrics


def test_regression_metrics_report_error_and_interval_scores_without_nll():
    mean = np.array([[1.0, 3.0], [4.0, 8.0]], dtype=np.float32)
    target = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    log_var = np.log(np.ones_like(mean, dtype=np.float32) * 4.0)

    metrics = regression_metrics(mean, log_var, target)

    assert set(metrics) == {
        "mae",
        "rmse",
        "nmae",
        "nrmse",
        "picp_90",
        "pinaw_90",
        "picp_95",
        "pinaw_95",
    }
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(14.0 / 4.0))
    assert metrics["nmae"] == pytest.approx(1.5 / 4.0)
    assert metrics["nrmse"] == pytest.approx(np.sqrt(14.0 / 4.0) / 4.0)
    assert metrics["picp_90"] == pytest.approx(1.0)
    assert metrics["pinaw_90"] == pytest.approx((2 * 1.6448536269514722 * 2.0) / 4.0)


def test_regression_metrics_can_use_fixed_normalization_scale():
    mean = np.array([[8.0, 12.0]], dtype=np.float32)
    target = np.array([[10.0, 10.0]], dtype=np.float32)
    log_var = np.log(np.ones_like(mean, dtype=np.float32))

    metrics = regression_metrics(mean, log_var, target, normalization_scale_value=20.0)

    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["nmae"] == pytest.approx(0.1)
    assert metrics["nrmse"] == pytest.approx(0.1)


def test_regression_metrics_mark_interval_scores_nan_without_uncertainty():
    mean = np.array([[1.0, 3.0]], dtype=np.float32)
    target = np.array([[2.0, 1.0]], dtype=np.float32)
    log_var = np.full_like(mean, np.nan, dtype=np.float32)

    metrics = regression_metrics(mean, log_var, target)

    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5.0 / 2.0))
    assert np.isnan(metrics["picp_90"])
    assert np.isnan(metrics["pinaw_90"])
    assert np.isnan(metrics["picp_95"])
    assert np.isnan(metrics["pinaw_95"])


def test_generation_period_metrics_include_0600_through_1800_targets():
    frame = pd.DataFrame(
        {
            "target_time": [
                "2020-01-01 05:45:00",
                "2020-01-01 06:00:00",
                "2020-01-01 18:00:00",
                "2020-01-01 18:15:00",
            ],
            "target": [100.0, 10.0, 20.0, 200.0],
            "mean": [0.0, 12.0, 23.0, 0.0],
            "log_var": np.log([1.0, 4.0, 9.0, 1.0]),
        }
    )

    metrics = generation_period_metrics(frame)

    assert metrics["mae"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt((2.0**2 + 3.0**2) / 2.0))
