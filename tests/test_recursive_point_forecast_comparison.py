from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pv import WindowArrays
from src.experiments.compare_recursive_point_forecasts_4h import (
    POINT_METRIC_NAMES,
    RecursivePointForecastRun,
    recursive_point_forecast_specs,
    run_recursive_point_forecast_comparison,
    write_point_forecast_artifacts,
)


class AddOneModel:
    def __call__(self, batch):
        mean = batch["direct"][:, :1] + 1.0
        return mean.astype(np.float32), np.full_like(mean, np.nan, dtype=np.float32)


def test_recursive_point_forecast_specs_compare_all_models_recursively():
    specs = recursive_point_forecast_specs()

    assert [spec.label for spec in specs] == ["BNN", "MLP", "1D-CNN", "LSTM"]
    assert all(spec.recursive for spec in specs)


def test_recursive_point_comparison_reuses_previous_predictions_for_each_model(tmp_path: Path, monkeypatch):
    arrays = WindowArrays(
        history=np.zeros((2, 3, 1), dtype=np.float32),
        weather=np.zeros((2, 4, 1), dtype=np.float32),
        direct=np.array([[1.0], [5.0]], dtype=np.float32),
        target=np.array([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]], dtype=np.float32),
        target_time=np.array(
            [
                ["2020-01-01 08:00", "2020-01-01 08:15", "2020-01-01 08:30", "2020-01-01 08:45"],
                ["2020-01-01 09:00", "2020-01-01 09:15", "2020-01-01 09:30", "2020-01-01 09:45"],
            ]
        ),
    )

    def fake_load_config(path):
        return {
            "output_dir": str(tmp_path / "outputs"),
            "data": {
                "lookback": 3,
                "horizon": 4,
                "features": {"history": ["AC_POWER"], "weather": ["IRRADIATION"], "direct": ["AC_POWER"], "target": "AC_POWER"},
            },
            "model": {"name": Path(path).stem},
            "training": {"backend": "numpy", "epochs": 1},
        }

    monkeypatch.setattr("src.experiments.compare_recursive_point_forecasts_4h.load_config", fake_load_config)
    monkeypatch.setattr("src.experiments.compare_recursive_point_forecasts_4h.load_or_make_split_arrays", lambda config: {"train": arrays, "val": arrays, "test": arrays})
    monkeypatch.setattr("src.experiments.compare_recursive_point_forecasts_4h.build_model", lambda config: AddOneModel())
    monkeypatch.setattr("src.experiments.compare_recursive_point_forecasts_4h.train_model", lambda model, split_arrays, config: [{"epoch": 1.0, "loss": 0.0}])

    out_dir = run_recursive_point_forecast_comparison(output_dir=tmp_path / "outputs", configs=["bnn.yaml", "mlp.yaml"])

    assert out_dir.parent == tmp_path / "outputs" / "comparisons"
    metrics_text = (out_dir / "model_metrics.csv").read_text(encoding="utf-8")
    for metric_name in POINT_METRIC_NAMES:
        assert metric_name in metrics_text
    assert "test_picp_90" not in metrics_text
    bnn_predictions = pd.read_csv(out_dir / "predictions" / "BNN.csv")
    first_sample = bnn_predictions[bnn_predictions["sample"] == 0].sort_values("horizon")
    assert first_sample["mean"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert (out_dir / "figures" / "prediction_0800_1200.png").read_bytes().startswith(b"\x89PNG")


def test_write_point_forecast_artifacts_writes_metric_pngs(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "label": ["MLP", "MLP"],
            "sample": [0, 0],
            "horizon": [0, 1],
            "target": [1.0, 2.0],
            "mean": [1.5, 2.5],
            "log_var": [np.nan, np.nan],
            "target_time": ["2020-01-01 08:00", "2020-01-01 08:15"],
        }
    )
    run = RecursivePointForecastRun(
        label="MLP",
        model="mlp_baseline",
        run_dir=tmp_path / "runs" / "mlp",
        predictions=frame,
        metrics={name: 0.1 for name in POINT_METRIC_NAMES},
        epoch_history=[{"epoch": 1.0, "loss": 0.5}],
        duration_seconds=1.0,
    )

    write_point_forecast_artifacts(tmp_path, [run], compare_config={"runs": []})

    assert (tmp_path / "figures" / "metrics_test_mae.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "figures" / "loss_curves.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "summary.md").is_file()
