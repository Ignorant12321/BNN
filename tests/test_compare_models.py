"""模型对比入口测试。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src import compare_models


def test_run_comparison_trains_all_configured_models(monkeypatch, tmp_path):
    """compare.yaml 中列出的 baseline 不应被跳过。"""
    config = {
        "seed": 42,
        "output_dir": str(tmp_path / "outputs"),
        "data": {"lookback": 4, "horizon": 2},
        "model": {"name": "improved_bnn", "hidden_dim": 8, "branch_dim": 4},
        "training": {"epochs": 1},
        "compare": {"models": ["improved_bnn", "mlp_baseline", "cnn_baseline", "mc_dropout"]},
    }
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    trained = []

    def fake_run_training(trial_config):
        model_name = trial_config["model"]["name"]
        trained.append(model_name)
        model_output_dir = Path(trial_config["output_dir"])
        assert model_output_dir.parent == tmp_path / "outputs" / "compare"
        assert re.fullmatch(r"\d{8}-\d{6}", model_output_dir.name)
        run_dir = tmp_path / "runs" / model_name
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "predictions").mkdir(parents=True)
        (run_dir / "metrics" / "metrics.json").write_text(
            json.dumps({"rmse": float(len(trained)), "mae": 0.5}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"sample": 0, "horizon": 1, "target_time": "2020-01-01 10:00:00", "y_true": 1.0, "y_mean": 1.0 + len(trained), "y_std": 0.1},
                {"sample": 0, "horizon": 2, "target_time": "2020-01-01 10:15:00", "y_true": 2.0, "y_mean": 2.0 + len(trained), "y_std": 0.1},
                {"sample": 1, "horizon": 1, "target_time": "2020-01-01 10:15:00", "y_true": 2.0, "y_mean": 2.0 + len(trained), "y_std": 0.1},
                {"sample": 1, "horizon": 2, "target_time": "2020-01-01 10:30:00", "y_true": 3.0, "y_mean": 3.0 + len(trained), "y_std": 0.1},
            ]
        ).to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)
        return run_dir

    monkeypatch.setattr(compare_models, "run_training", fake_run_training)

    out_dir = compare_models.run_comparison(config_path)

    assert trained == ["improved_bnn", "mlp_baseline", "cnn_baseline", "mc_dropout"]
    assert out_dir.parent == tmp_path / "outputs" / "compare"
    assert re.fullmatch(r"\d{8}-\d{6}", out_dir.name)
    rows = pd.read_csv(out_dir / "model_metrics.csv")
    assert rows["model"].tolist() == trained
    assert (out_dir / "figures" / "compare_prediction_mean.png").is_file()
    assert (out_dir / "figures" / "compare_horizon_rmse.png").is_file()
    assert (out_dir / "figures" / "compare_metrics.png").is_file()
