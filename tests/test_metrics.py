import numpy as np
import pytest

from src.evaluation.metrics import regression_metrics


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
