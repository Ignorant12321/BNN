from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pv import WindowArrays
from src.experiments.compare_bnn_strategies_4h import (
    DEFAULT_BNN_4H_CONFIG,
    combine_direct_prediction_frames,
    make_direct_step_config,
    make_recursive_step_config,
    recursive_forecast_config,
    recursive_prediction_frame,
    slice_step_arrays,
)


class OneStepEchoModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        mean = batch["direct"][:, :1] + self.calls
        log_var = np.full_like(mean, -2.0, dtype=np.float32)
        return mean.astype(np.float32), log_var


def test_slice_step_arrays_keeps_inputs_and_selects_one_forecast_step():
    arrays = WindowArrays(
        history=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        weather=np.arange(16, dtype=np.float32).reshape(2, 4, 2),
        direct=np.array([[1.0], [2.0]], dtype=np.float32),
        target=np.array([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]], dtype=np.float32),
        target_time=np.array(
            [
                ["2020-01-01 00:15", "2020-01-01 00:30", "2020-01-01 00:45", "2020-01-01 01:00"],
                ["2020-01-01 01:15", "2020-01-01 01:30", "2020-01-01 01:45", "2020-01-01 02:00"],
            ]
        ),
    )

    step = slice_step_arrays(arrays, 2)

    np.testing.assert_array_equal(step.history, arrays.history)
    np.testing.assert_array_equal(step.direct, arrays.direct)
    np.testing.assert_array_equal(step.weather, arrays.weather[:, 2:3, :])
    np.testing.assert_array_equal(step.target, np.array([[12.0], [22.0]], dtype=np.float32))
    np.testing.assert_array_equal(step.target_time, arrays.target_time[:, 2:3])


def test_slice_step_arrays_can_select_multiple_forecast_steps():
    arrays = WindowArrays(
        history=np.zeros((1, 2, 1), dtype=np.float32),
        weather=np.arange(5, dtype=np.float32).reshape(1, 5, 1),
        direct=np.array([[1.0]], dtype=np.float32),
        target=np.array([[10.0, 11.0, 12.0, 13.0, 14.0]], dtype=np.float32),
    )

    step = slice_step_arrays(arrays, 1, step_count=3)

    np.testing.assert_array_equal(step.weather, np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32))
    np.testing.assert_array_equal(step.target, np.array([[11.0, 12.0, 13.0]], dtype=np.float32))


def test_combine_direct_prediction_frames_restores_four_hour_horizon_order():
    frames = [
        pd.DataFrame({"label": ["step"], "sample": [1], "horizon": [0], "target": [21.0], "mean": [20.0], "log_var": [0.2]}),
        pd.DataFrame({"label": ["step"], "sample": [0], "horizon": [0], "target": [10.0], "mean": [9.0], "log_var": [0.1]}),
    ]

    combined = combine_direct_prediction_frames(frames, label="Direct")

    assert combined["label"].tolist() == ["Direct", "Direct"]
    assert combined["sample"].tolist() == [0, 1]
    assert combined["horizon"].tolist() == [1, 0]
    assert combined["mean"].tolist() == [9.0, 20.0]


def test_make_direct_step_config_does_not_mutate_four_hour_config():
    config = {"data": {"horizon": 16}, "model": {"name": "improved_bnn"}}

    step_config = make_direct_step_config(config, 0)

    assert step_config["data"]["horizon"] == 1
    assert config["data"]["horizon"] == 16


def test_make_recursive_step_config_uses_strategy_train_horizon():
    config = {"data": {"horizon": 16}, "strategy": {"name": "recursive", "train_horizon": 3, "forecast_horizon": 16}}

    step_config = make_recursive_step_config(config)

    assert step_config["data"]["horizon"] == 3
    assert config["data"]["horizon"] == 16


def test_recursive_forecast_config_uses_strategy_forecast_horizon():
    config = {"data": {"horizon": 16}, "strategy": {"name": "recursive", "train_horizon": 1, "forecast_horizon": 8}}

    forecast_config = recursive_forecast_config(config)

    assert forecast_config["data"]["horizon"] == 8
    assert config["data"]["horizon"] == 16


def test_default_comparison_config_is_bnn_four_hour_yaml():
    assert DEFAULT_BNN_4H_CONFIG == Path("configs/models/bnn/4h.yaml")


def test_recursive_prediction_frame_feeds_previous_prediction_forward():
    arrays = WindowArrays(
        history=np.zeros((2, 3, 1), dtype=np.float32),
        weather=np.zeros((2, 4, 2), dtype=np.float32),
        direct=np.array([[10.0], [20.0]], dtype=np.float32),
        target=np.array([[11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0]], dtype=np.float32),
        target_time=np.array(
            [
                ["2020-01-01 00:15", "2020-01-01 00:30", "2020-01-01 00:45", "2020-01-01 01:00"],
                ["2020-01-01 01:15", "2020-01-01 01:30", "2020-01-01 01:45", "2020-01-01 02:00"],
            ]
        ),
    )

    frame = recursive_prediction_frame(
        label="Recursive",
        model=OneStepEchoModel(),
        arrays=arrays,
        config={"data": {"horizon": 4}},
    )

    first_sample = frame[frame["sample"] == 0].sort_values("horizon")
    assert first_sample["mean"].tolist() == [11.0, 13.0, 16.0, 20.0]
    assert first_sample["horizon"].tolist() == [0, 1, 2, 3]
