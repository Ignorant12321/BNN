"""模型对比入口测试。"""

from __future__ import annotations

import json

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
        run_dir = tmp_path / "runs" / model_name
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "metrics" / "metrics.json").write_text(
            json.dumps({"rmse": float(len(trained)), "mae": 0.5}),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(compare_models, "run_training", fake_run_training)

    out_dir = compare_models.run_comparison(config_path)

    assert trained == ["improved_bnn", "mlp_baseline", "cnn_baseline", "mc_dropout"]
    rows = pd.read_csv(out_dir / "model_metrics.csv")
    assert rows["model"].tolist() == trained
