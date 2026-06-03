from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.pv import WindowArrays
from src.experiments.compare_recursive_interval_methods_4h import (
    build_coverage_rows,
    interval_coverage,
    interval_pinaw,
    normal_residual_intervals,
    persistence_interval_frame,
    write_coverage_summary,
)


def test_interval_coverage_reports_percent_inside_interval():
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 5.0, 9.0],
            "lower_90": [0.0, 2.0, 4.0, 10.0],
            "upper_90": [1.0, 3.0, 6.0, 11.0],
        }
    )

    assert interval_coverage(frame, "lower_90", "upper_90") == pytest.approx(75.0)


def test_interval_pinaw_reports_normalized_mean_width():
    frame = pd.DataFrame(
        {
            "target": [0.0, 10.0],
            "lower_90": [1.0, 2.0],
            "upper_90": [5.0, 8.0],
        }
    )

    assert interval_pinaw(frame, "lower_90", "upper_90") == pytest.approx(0.5)


def test_normal_residual_intervals_calibrate_by_horizon():
    val = pd.DataFrame(
        {
            "horizon": [0, 0, 1, 1],
            "target": [10.0, 12.0, 20.0, 24.0],
            "mean": [9.0, 9.0, 22.0, 22.0],
        }
    )
    test = pd.DataFrame(
        {
            "horizon": [0, 1],
            "target": [0.0, 0.0],
            "mean": [100.0, 200.0],
        }
    )

    intervals = normal_residual_intervals(val, test, levels=(90,))

    assert intervals.loc[0, "lower_90"] == pytest.approx(102.0 - 1.6448536269514722)
    assert intervals.loc[0, "upper_90"] == pytest.approx(102.0 + 1.6448536269514722)
    assert intervals.loc[1, "lower_90"] == pytest.approx(200.0 - 1.6448536269514722 * 2.0)
    assert intervals.loc[1, "upper_90"] == pytest.approx(200.0 + 1.6448536269514722 * 2.0)


def test_persistence_interval_frame_uses_direct_power_and_validation_quantiles():
    val_arrays = WindowArrays(
        history=np.zeros((3, 1, 1), dtype=np.float32),
        weather=np.zeros((3, 2, 1), dtype=np.float32),
        direct=np.array([[10.0], [10.0], [10.0]], dtype=np.float32),
        target=np.array([[8.0, 12.0], [10.0, 14.0], [12.0, 16.0]], dtype=np.float32),
    )
    test_arrays = WindowArrays(
        history=np.zeros((1, 1, 1), dtype=np.float32),
        weather=np.zeros((1, 2, 1), dtype=np.float32),
        direct=np.array([[20.0]], dtype=np.float32),
        target=np.array([[19.0, 30.0]], dtype=np.float32),
    )

    frame = persistence_interval_frame(val_arrays, test_arrays, levels=(90,))

    assert list(frame["mean"]) == [20.0, 20.0]
    assert frame.loc[0, "lower_90"] == pytest.approx(18.2)
    assert frame.loc[0, "upper_90"] == pytest.approx(21.8)
    assert frame.loc[1, "lower_90"] == pytest.approx(22.2)
    assert frame.loc[1, "upper_90"] == pytest.approx(25.8)


def test_write_coverage_summary_creates_comparison_csv(tmp_path: Path):
    our = pd.DataFrame(
        {
            "target": [1.0, 2.0],
            "lower_90": [0.0, 0.0],
            "upper_90": [2.0, 3.0],
            "lower_95": [0.0, 0.0],
            "upper_95": [2.0, 1.0],
        }
    )
    normal = our.copy()
    persistence = our.copy()

    rows = build_coverage_rows(our, normal, persistence, levels=(90, 95))
    write_coverage_summary(tmp_path / "coverage_summary.csv", rows)

    text = (tmp_path / "coverage_summary.csv").read_text(encoding="utf-8")
    assert (
        "confidence,our_method_picp,our_method_pinaw,normal_picp,normal_pinaw,"
        "persistence_picp,persistence_pinaw"
    ) in text
    assert "90,100.0,2.5,100.0,2.5,100.0,2.5" in text
    assert "95,50.0,1.5,50.0,1.5,50.0,1.5" in text
