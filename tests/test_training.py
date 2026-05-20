import numpy as np
import pandas as pd
import pytest

from src.data.pv import WindowArrays
import src.experiments.train as train_module
from src.experiments.train import evaluate_model, load_or_make_split_arrays, run_training
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
    assert run_dir.parent.parent.name == "runs"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "manifest.json").is_file()
    train_history = (run_dir / "metrics" / "train_history.csv").read_text(encoding="utf-8")
    assert "train_rmse" in train_history
    assert "val_rmse" in train_history
    assert "test_rmse" not in train_history
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "models" / "best.pkl").is_file()
    output = capsys.readouterr().out
    assert output.index("Training Parameters") < output.index("Training Process") < output.index("Training Results")
    assert "Start Time" in output
    assert "Fit" in output
    assert "Duration" in output
    assert "Val RMSE" in output
    assert str(run_dir) in output
