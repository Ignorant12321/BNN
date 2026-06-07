from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.pv import WindowArrays
from src.experiments import train_bnn_recursive_4h


def test_recursive_four_hour_experiment_uses_recursive_config_by_default():
    assert train_bnn_recursive_4h.DEFAULT_RECURSIVE_BNN_4H_CONFIG == Path("configs/models/bnn/recursive_4h.yaml")


def test_recursive_train_run_dir_uses_train_model_folder(tmp_path: Path):
    run_dir = train_bnn_recursive_4h.make_recursive_train_run_dir(tmp_path)

    assert run_dir.parent.name == "improved_bnn_recursive"
    assert run_dir.parent.parent.name == "train"


def test_recursive_train_run_dir_can_use_custom_model_folder(tmp_path: Path):
    run_dir = train_bnn_recursive_4h.make_recursive_train_run_dir(tmp_path, model_folder="pv_usibnn_recursive")

    assert run_dir.parent.name == "pv_usibnn_recursive"
    assert run_dir.parent.parent.name == "train"


def test_recursive_strategy_model_name_preserves_explicit_recursive_model():
    config = {"model": {"name": "pv_usibnn_recursive"}}

    assert train_bnn_recursive_4h.recursive_strategy_model_name(config) == "pv_usibnn_recursive"


def test_recursive_saved_config_keeps_model_loadable_and_records_forecast_horizon():
    forecast_config = {"data": {"horizon": 16}, "strategy": {"name": "recursive"}}
    step_config = {"data": {"horizon": 1}, "model": {"name": "improved_bnn"}}

    saved = train_bnn_recursive_4h.recursive_saved_config(forecast_config, step_config)

    assert saved["model"]["name"] == "improved_bnn"
    assert saved["data"]["horizon"] == 1
    assert saved["strategy"]["train_horizon"] == 1
    assert saved["strategy"]["forecast_horizon"] == 16


def test_recursive_training_early_stopping_uses_full_recursive_validation(monkeypatch, tmp_path: Path):
    arrays = WindowArrays(
        history=np.zeros((2, 2, 1), dtype=np.float32),
        weather=np.zeros((2, 4, 1), dtype=np.float32),
        direct=np.zeros((2, 1), dtype=np.float32),
        target=np.ones((2, 4), dtype=np.float32),
        target_time=np.array(
            [
                ["2020-01-01 08:00:00", "2020-01-01 08:15:00", "2020-01-01 08:30:00", "2020-01-01 08:45:00"],
                ["2020-01-01 09:00:00", "2020-01-01 09:15:00", "2020-01-01 09:30:00", "2020-01-01 09:45:00"],
            ]
        ),
    )
    validation_horizons = []

    monkeypatch.setattr(train_bnn_recursive_4h, "load_or_make_split_arrays", lambda config: {"train": arrays, "val": arrays, "test": arrays})
    monkeypatch.setattr(train_bnn_recursive_4h, "build_model", lambda config: object())
    monkeypatch.setattr(train_bnn_recursive_4h, "should_scale_torch_training", lambda config: False)
    monkeypatch.setattr(train_bnn_recursive_4h, "print_training_parameters", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "print_training_process_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "print_training_process_end", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "print_recursive_training_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "save_model_artifact", lambda *args, **kwargs: tmp_path / "models" / "best.pt")
    monkeypatch.setattr(train_bnn_recursive_4h, "save_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "save_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "write_recursive_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_bnn_recursive_4h, "write_recursive_log", lambda *args, **kwargs: None)

    def fake_recursive_prediction_frame(label, model, arrays_arg, config):
        validation_horizons.append(arrays_arg.target.shape[1])
        return pd.DataFrame(
            {
                "label": [label],
                "sample": [0],
                "horizon": [0],
                "target": [1.0],
                "mean": [1.0],
                "log_var": [-2.0],
                "target_time": ["2020-01-01 08:00:00"],
            }
        )

    def fake_train_model(model, arrays_by_split, config, epoch_callback=None, validation_metrics_callback=None):
        assert arrays_by_split["val"].target.shape[1] == 1
        assert validation_metrics_callback is not None
        metrics = validation_metrics_callback(model)
        assert "val_generation_nrmse" in metrics
        return SimpleNamespace(metrics=metrics, epoch_history=[{"epoch": 1.0, **metrics}], best_epoch=1)

    monkeypatch.setattr(train_bnn_recursive_4h, "recursive_prediction_frame", fake_recursive_prediction_frame)
    monkeypatch.setattr(train_bnn_recursive_4h, "train_model", fake_train_model)

    config = {
        "seed": 1,
        "data": {"horizon": 4},
        "strategy": {"name": "recursive", "train_horizon": 1, "forecast_horizon": 4},
        "model": {"name": "pv_usibnn_recursive"},
        "training": {"early_stopping": {"metric": "val_generation_nrmse"}},
    }

    train_bnn_recursive_4h.run_recursive_training(config, run_dir=tmp_path)

    assert validation_horizons[0] == 4


def test_recursive_prediction_metrics_include_generation_split():
    predictions = pd.DataFrame(
        {
            "target": [0.0, 10.0, 10.0],
            "mean": [0.0, 8.0, 7.0],
            "log_var": [-2.0, -2.0, -2.0],
            "target_time": ["2020-01-01 05:00:00", "2020-01-01 08:00:00", "2020-01-01 19:00:00"],
        }
    )

    metrics = train_bnn_recursive_4h.recursive_prediction_metrics("val", predictions)

    assert metrics["val_mae"] == pytest.approx(5.0 / 3.0)
    assert metrics["val_generation_mae"] == pytest.approx(2.0)
    assert "val_generation_nrmse" in metrics


def test_write_recursive_outputs_creates_training_style_files(tmp_path: Path):
    run = train_bnn_recursive_4h.RecursiveExperimentResult(
        run_dir=tmp_path / "recursive",
        metrics={"test_mae": 1.0, "test_rmse": 2.0},
        predictions=pd.DataFrame(
            {
                "label": ["Recursive"],
                "sample": [0],
                "horizon": [0],
                "target": [1.0],
                "mean": [1.1],
                "log_var": [-2.0],
            }
        ),
        epoch_history=[{"epoch": 1.0, "loss": 0.5}],
        model_path=tmp_path / "recursive" / "models" / "best.pt",
    )

    train_bnn_recursive_4h.write_recursive_outputs(tmp_path, run)

    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "epoch_history.csv").is_file()
    assert (tmp_path / "predictions" / "test.csv").is_file()
    assert (tmp_path / "figures" / "loss_curve.png").is_file()
    assert (tmp_path / "figures" / "prediction_window_metrics.csv").is_file()


def test_print_recursive_training_results_includes_training_sections(capsys, tmp_path: Path):
    run = train_bnn_recursive_4h.RecursiveExperimentResult(
        run_dir=tmp_path,
        metrics={"test_mae": 1.0, "test_rmse": 2.0},
        predictions=pd.DataFrame(),
        epoch_history=[{"epoch": 1.0, "loss": 0.5}],
        model_path=tmp_path / "models" / "best.pt",
        duration_seconds=1.25,
    )

    train_bnn_recursive_4h.print_recursive_training_results(run)

    output = capsys.readouterr().out
    assert "Training Results" in output
    assert "Test MAE" in output
    assert str(run.run_dir) in output
