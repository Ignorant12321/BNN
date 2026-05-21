import math

import pandas as pd
import pytest

from src.evaluation.plots import prediction_interval_bounds


def test_prediction_interval_bounds_combines_duplicate_target_times():
    frame = pd.DataFrame(
        {
            "label": ["BNN", "BNN", "BNN"],
            "target_time": ["2020-01-01 08:00:00", "2020-01-01 08:00:00", "2020-01-01 08:15:00"],
            "mean": [8.0, 12.0, 20.0],
            "log_var": [math.log(4.0), math.log(4.0), math.log(9.0)],
        }
    )

    bounds = prediction_interval_bounds(frame, "BNN")

    first = bounds.iloc[0]
    assert first["mean"] == pytest.approx(10.0)
    assert first["std"] == pytest.approx(math.sqrt(8.0))
    assert first["lower_90"] == pytest.approx(10.0 - 1.6448536269514722 * math.sqrt(8.0))
    assert first["upper_90"] == pytest.approx(10.0 + 1.6448536269514722 * math.sqrt(8.0))
    assert first["lower_95"] == pytest.approx(10.0 - 1.959963984540054 * math.sqrt(8.0))
    assert first["upper_95"] == pytest.approx(10.0 + 1.959963984540054 * math.sqrt(8.0))
    second = bounds.iloc[1]
    assert second["std"] == pytest.approx(3.0)
