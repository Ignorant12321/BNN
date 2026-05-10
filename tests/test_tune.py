"""Optuna 调参目标测试。"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from src import tune
from src.tune import export_study_results, load_objective_metric, merge_best_params, prepare_trial_config


def test_load_objective_metric_reads_validation_metrics(tmp_path):
    """调参目标必须来自验证集，不能读取测试集指标。"""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "validation_metrics.json").write_text(json.dumps({"rmse": 1.5}), encoding="utf-8")
    (metrics_dir / "metrics.json").write_text(json.dumps({"rmse": 0.1}), encoding="utf-8")

    assert load_objective_metric(tmp_path, metric="rmse") == 1.5


def test_prepare_trial_config_disables_test_evaluation():
    """Optuna trial 期间不应反复评估测试集。"""
    config = prepare_trial_config({"model": {}, "training": {}})

    assert config["evaluation"]["run_test"] is False


def test_prepare_trial_config_routes_trial_outputs_to_tuning_run(tmp_path):
    """Optuna trial 输出应放到对应调参会话目录，避免污染正式训练目录。"""
    config = prepare_trial_config(
        {"output_dir": str(tmp_path), "model": {}, "training": {}},
        tuning_run_name="fixed",
    )

    assert config["output_dir"] == str(tmp_path / "tuning" / "fixed")


def test_prepare_trial_config_disables_workers_on_windows():
    """Windows CUDA tuning should avoid spawning extra DataLoader worker processes."""
    config = prepare_trial_config(
        {"model": {}, "training": {"num_workers": 2, "persistent_workers": True}},
        platform="win32",
    )

    assert config["training"]["num_workers"] == 0
    assert config["training"]["persistent_workers"] is False


def test_create_tuning_study_uses_configured_persistent_storage(tmp_path):
    """Optuna study should use configured storage so interrupted tuning can resume."""
    assert hasattr(tune, "create_tuning_study"), "create_tuning_study helper is missing"

    calls = []

    class FakeOptuna:
        @staticmethod
        def create_study(**kwargs):
            calls.append(kwargs)
            return "study"

    base_config = {
        "output_dir": str(tmp_path),
        "tuning": {
            "study_name": "solar-bnn",
            "storage": f"sqlite:///{tmp_path / 'optuna.db'}",
            "load_if_exists": True,
        },
    }

    study = tune.create_tuning_study(FakeOptuna, base_config)

    assert study == "study"
    assert calls == [
        {
            "direction": "minimize",
            "study_name": "solar-bnn",
            "storage": f"sqlite:///{tmp_path / 'optuna.db'}",
            "load_if_exists": True,
        }
    ]


def test_count_remaining_trials_treats_n_trials_as_total_target():
    """Resumed tuning should add only the trials needed to reach the configured total."""
    assert hasattr(tune, "count_remaining_trials"), "count_remaining_trials helper is missing"

    class FakeStudy:
        trials = [object(), object(), object()]

    assert tune.count_remaining_trials(FakeStudy(), target_n_trials=5) == 2
    assert tune.count_remaining_trials(FakeStudy(), target_n_trials=3) == 0


def test_merge_best_params_updates_model_and_training_values():
    """最佳参数应能并入正式训练配置。"""
    config = {
        "model": {"hidden_dim": 128, "branch_dim": 64},
        "training": {"lr": 0.001, "kl_beta": 0.0001},
    }
    best_params = {
        "hidden_dim": 256,
        "branch_dim": 64,
        "lr": 0.00019544293665253097,
        "kl_beta": 1.2114960587728605e-05,
    }

    merged = merge_best_params(config, best_params)

    assert merged["model"]["hidden_dim"] == 256
    assert merged["model"]["branch_dim"] == 64
    assert merged["training"]["lr"] == 0.00019544293665253097
    assert merged["training"]["kl_beta"] == 1.2114960587728605e-05
    assert config["model"]["hidden_dim"] == 128


def test_export_study_results_writes_summary_trials_and_best_config(tmp_path):
    """调参结束后应导出可复用的结果文件。"""

    class FakeStudy:
        best_params = {
            "hidden_dim": 256,
            "branch_dim": 64,
            "lr": 0.00019544293665253097,
            "kl_beta": 1.2114960587728605e-05,
        }
        best_value = 2448.0612226049443

        class BestTrial:
            number = 10

        best_trial = BestTrial()

        def trials_dataframe(self):
            return pd.DataFrame(
                [
                    {"number": 10, "value": 2448.0612226049443, "params_hidden_dim": 256},
                    {"number": 19, "value": 2604.117604366335, "params_hidden_dim": 128},
                ]
            )

    base_config = {
        "output_dir": str(tmp_path),
        "model": {"name": "improved_bnn", "hidden_dim": 128, "branch_dim": 64},
        "training": {"lr": 0.001, "kl_beta": 0.0001},
    }

    export_dir = export_study_results(FakeStudy(), base_config, run_name="fixed")

    assert export_dir == tmp_path / "tuning" / "fixed"
    summary = json.loads((export_dir / "best_params.json").read_text(encoding="utf-8"))
    assert summary["best_trial"] == 10
    assert summary["best_value"] == 2448.0612226049443
    assert summary["best_params"]["hidden_dim"] == 256
    assert (export_dir / "trials.csv").exists()
    best_config = yaml.safe_load((export_dir / "best_config.yaml").read_text(encoding="utf-8"))
    assert best_config["model"]["hidden_dim"] == 256
    assert best_config["training"]["kl_beta"] == 1.2114960587728605e-05
