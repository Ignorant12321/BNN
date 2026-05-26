import numpy as np
import pandas as pd
import pytest

from src.data.pv import WindowArrays
import src.experiments.train as train_module
from src.experiments.train import evaluate_model, load_or_make_split_arrays, run_training
from src.training.trainer import best_epoch_from_history
from src.models.baselines import RidgeProbabilisticModel


class RowIndexModel(RidgeProbabilisticModel):
    def __init__(self):
        super().__init__(horizon=1)

    def features_from_batch(self, batch):
        return batch["direct"].astype(np.float32)

    def __call__(self, batch):
        mean = batch["direct"].astype(np.float32)
        log_var = np.zeros_like(mean, dtype=np.float32)
        return mean, log_var


def test_evaluate_model_uses_all_windows_not_only_first_batch(tmp_path):
    arrays = WindowArrays(
        history=np.zeros((3, 1, 1), dtype=np.float32),
        weather=np.zeros((3, 1, 1), dtype=np.float32),
        direct=np.array([[0.0], [10.0], [20.0]], dtype=np.float32),
        target=np.array([[0.0], [0.0], [0.0]], dtype=np.float32),
    )

    metrics = evaluate_model(RowIndexModel(), arrays)

    assert metrics["rmse"] == pytest.approx(np.sqrt((0.0**2 + 10.0**2 + 20.0**2) / 3.0))


def test_best_epoch_from_history_uses_configured_monitor_metric():
    history = [
        {"epoch": 1.0, "loss": 3.0, "val_rmse": 1.0, "val_generation_nrmse": 0.30},
        {"epoch": 2.0, "loss": 2.0, "val_rmse": 2.0, "val_generation_nrmse": 0.10},
    ]

    assert best_epoch_from_history(history, monitor_metric="val_generation_nrmse") == 2


def test_load_or_make_split_arrays_reads_existing_splits_without_cross_split_history(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    base = {
        "AMBIENT_TEMPERATURE": 25.0,
        "MODULE_TEMPERATURE": 30.0,
        "IRRADIATION": 0.5,
    }
    for split_name, values in {
        "train": [1.0, 2.0, 3.0],
        "val": [100.0, 101.0, 102.0],
        "test": [200.0, 201.0, 202.0],
    }.items():
        frame = pd.DataFrame(
            {
                "DATE_TIME": pd.date_range("2020-01-01", periods=3, freq="15min"),
                "AC_POWER": values,
                **base,
            }
        )
        frame.to_csv(processed_dir / f"{split_name}.csv", index=False)
    config = {
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 2,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        }
    }

    arrays_by_split = load_or_make_split_arrays(config)

    np.testing.assert_array_equal(arrays_by_split["train"].history[:, :, 0], np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays_by_split["train"].target, np.array([[3.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays_by_split["val"].history[:, :, 0], np.array([[100.0, 101.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays_by_split["val"].target, np.array([[102.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays_by_split["test"].history[:, :, 0], np.array([[200.0, 201.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays_by_split["test"].target, np.array([[202.0]], dtype=np.float32))


def test_run_training_fits_only_train_split_and_writes_artifacts(tmp_path, monkeypatch, capsys):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for split_name, values in {
        "train": [1.0, 2.0, 3.0],
        "val": [100.0, 101.0, 102.0],
        "test": [200.0, 201.0, 202.0],
    }.items():
        frame = pd.DataFrame(
            {
                "DATE_TIME": pd.date_range("2020-01-01", periods=3, freq="15min"),
                "AC_POWER": values,
                "AMBIENT_TEMPERATURE": 25.0,
                "MODULE_TEMPERATURE": 30.0,
                "IRRADIATION": 0.5,
            }
        )
        frame.to_csv(processed_dir / f"{split_name}.csv", index=False)

    class RecordingModel:
        fitted_target = None

        def fit(self, arrays):
            self.fitted_target = arrays.target.copy()

        def __call__(self, batch):
            return np.zeros((len(batch["direct"]), 1), dtype=np.float32), np.zeros((len(batch["direct"]), 1), dtype=np.float32)

    model = RecordingModel()
    monkeypatch.setattr(train_module, "build_model", lambda config: model)
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 2,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "mlp_baseline"},
    }

    run_dir = run_training(config)

    np.testing.assert_array_equal(model.fitted_target, np.array([[3.0]], dtype=np.float32))
    assert run_dir.parent.parent.name == "train"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "manifest.json").is_file()
    epoch_history = (run_dir / "epoch_history.csv").read_text(encoding="utf-8")
    split_metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "epoch,loss" in epoch_history
    assert "train,rmse" in split_metrics
    assert "train,mae" in split_metrics
    assert "train_generation,rmse" in split_metrics
    assert "train_generation,picp_90" in split_metrics
    assert "val,rmse" in split_metrics
    assert "val,picp_90" in split_metrics
    assert "val_generation,rmse" in split_metrics
    assert "val_generation,picp_90" in split_metrics
    assert "test,rmse" in split_metrics
    assert "test,picp_90" in split_metrics
    assert "test_generation,rmse" in split_metrics
    assert "test_generation,picp_90" in split_metrics
    assert "nll" not in split_metrics
    loss_curve_path = run_dir / "figures" / "loss_curve.png"
    assert loss_curve_path.is_file()
    assert loss_curve_path.read_bytes().startswith(b"\x89PNG")
    prediction_path = run_dir / "predictions" / "test.csv"
    assert prediction_path.is_file()
    prediction_text = prediction_path.read_text(encoding="utf-8")
    assert "label,sample,horizon,target,mean,log_var,std,lower_90,upper_90,lower_95,upper_95,target_time" in prediction_text
    interval_metrics_path = run_dir / "figures" / "prediction_window_metrics.csv"
    assert interval_metrics_path.is_file()
    interval_metrics_text = interval_metrics_path.read_text(encoding="utf-8")
    assert "interval_start,interval_end,mae,rmse,nmae,nrmse,picp_90,pinaw_90,picp_95,pinaw_95" in interval_metrics_text
    for name in ("prediction_0800_1200.png", "prediction_1000_1400.png", "prediction_1200_1600.png"):
        prediction_figure = run_dir / "figures" / name
        assert prediction_figure.is_file()
        assert prediction_figure.read_bytes().startswith(b"\x89PNG")
    assert (run_dir / "train.log").is_file()
    assert not (run_dir / "logs").exists()
    assert not (run_dir / "metrics").exists()
    assert not (run_dir / "split_metrics.csv").exists()
    assert (run_dir / "models" / "best.pkl").is_file()
    output = capsys.readouterr().out
    assert output.index("Training Parameters") < output.index("Training Process") < output.index("Training Results")
    assert "Start Time" in output
    assert "Fit" in output
    assert "Duration" in output
    assert "Val RMSE" in output
    assert "Val NLL" not in output
    assert str(run_dir) in output
    log_text = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "Training Parameters" in log_text
    assert "Training Process" in log_text
    assert "Training Results" in log_text
    assert "Val RMSE" in log_text
    assert "Test RMSE" in log_text
    note_text = (run_dir / "note.txt").read_text(encoding="utf-8")
    assert note_text == f"{run_dir.name}\n"


def test_run_training_writes_note_file_from_argument(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for split_name, values in {
        "train": [1.0, 2.0, 3.0],
        "val": [100.0, 101.0, 102.0],
        "test": [200.0, 201.0, 202.0],
    }.items():
        frame = pd.DataFrame(
            {
                "DATE_TIME": pd.date_range("2020-01-01", periods=3, freq="15min"),
                "AC_POWER": values,
                "AMBIENT_TEMPERATURE": 25.0,
                "MODULE_TEMPERATURE": 30.0,
                "IRRADIATION": 0.5,
            }
        )
        frame.to_csv(processed_dir / f"{split_name}.csv", index=False)

    class RecordingModel:
        def fit(self, arrays):
            pass

        def __call__(self, batch):
            return np.zeros((len(batch["direct"]), 1), dtype=np.float32), np.zeros((len(batch["direct"]), 1), dtype=np.float32)

    monkeypatch.setattr(train_module, "build_model", lambda config: RecordingModel())
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 2,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "mlp_baseline"},
    }

    run_dir = run_training(config, note="first bnn trial")

    assert (run_dir / "note.txt").read_text(encoding="utf-8") == "first bnn trial\n"


def test_torch_training_batches_include_target_for_recursive_teacher_forcing():
    torch = pytest.importorskip("torch")
    from torch import nn

    from src.training.torch_trainer import train_torch_model

    class TargetAwareTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))
            self.saw_target = False

        def forward(self, batch):
            assert "target" in batch
            self.saw_target = True
            mean = torch.zeros_like(batch["target"]) + self.bias
            log_var = torch.zeros_like(mean)
            return mean, log_var

        def kl_loss(self):
            return self.bias * 0.0

    arrays = WindowArrays(
        history=np.ones((2, 2, 1), dtype=np.float32),
        weather=np.ones((2, 1, 1), dtype=np.float32),
        direct=np.ones((2, 1), dtype=np.float32),
        target=np.ones((2, 1), dtype=np.float32),
    )
    model = TargetAwareTorchModel()
    config = {"training": {"device": "cpu", "epochs": 1, "batch_size": 2, "lr": 0.01, "kl_beta": 0.0}}

    train_torch_model(model, arrays, config)

    assert model.saw_target
