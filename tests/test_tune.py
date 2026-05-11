"""Optuna 调参目标测试。"""

from __future__ import annotations

import inspect
import json

import pandas as pd
import yaml

from src import tune
from src.tune import (
    apply_trial_suggestions,
    export_study_results,
    format_duration,
    format_study_summary,
    format_trial_config,
    format_trial_result,
    load_validation_metrics,
    load_objective_metric,
    merge_best_params,
    prepare_trial_config,
    resolve_objective_metric,
)


def test_load_objective_metric_reads_validation_metrics(tmp_path):
    """调参目标必须来自验证集，不能读取测试集指标。"""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "validation_metrics.json").write_text(json.dumps({"rmse": 1.5}), encoding="utf-8")
    (metrics_dir / "metrics.json").write_text(json.dumps({"rmse": 0.1}), encoding="utf-8")

    assert load_objective_metric(tmp_path, metric="rmse") == 1.5


def test_load_validation_metrics_reads_full_validation_metrics(tmp_path):
    """Trial 结束摘要需要读取完整验证集指标。"""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "validation_metrics.json").write_text(
        json.dumps({"rmse": 1.5, "crps": 0.25}),
        encoding="utf-8",
    )

    assert load_validation_metrics(tmp_path) == {"rmse": 1.5, "crps": 0.25}


def test_resolve_objective_metric_defaults_to_rmse_and_accepts_crps():
    """调参目标可配置；未配置时保持历史 RMSE 行为。"""
    assert resolve_objective_metric({}) == "rmse"
    assert resolve_objective_metric({"tuning": {"objective_metric": "crps"}}) == "crps"


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


def test_apply_trial_suggestions_uses_configured_search_space():
    """Optuna 搜索范围应来自 tuning.yaml，而不是硬编码在调参入口中。"""

    class FakeTrial:
        def __init__(self):
            self.categorical_calls = []
            self.float_calls = []

        def suggest_categorical(self, name, choices):
            self.categorical_calls.append((name, choices))
            return choices[-1]

        def suggest_float(self, name, low, high, log=False):
            self.float_calls.append((name, low, high, log))
            return high

    config = {
        "model": {},
        "training": {},
        "tuning": {
            "search_space": {
                "hidden_dim": [128, 256],
                "branch_dim": [32, 64, 128],
                "lr": {"low": 5e-4, "high": 2.5e-3, "log": True},
                "kl_beta": {"low": 1e-5, "high": 3e-4, "log": True},
            }
        },
    }
    trial = FakeTrial()

    apply_trial_suggestions(config, trial)

    assert trial.categorical_calls == [
        ("hidden_dim", [128, 256]),
        ("branch_dim", [32, 64, 128]),
    ]
    assert trial.float_calls == [
        ("lr", 5e-4, 2.5e-3, True),
        ("kl_beta", 1e-5, 3e-4, True),
    ]
    assert config["model"]["hidden_dim"] == 256
    assert config["model"]["branch_dim"] == 128
    assert config["training"]["lr"] == 2.5e-3
    assert config["training"]["kl_beta"] == 3e-4


def test_format_trial_config_prints_one_based_trial_number_and_sampled_params():
    """控制台展示的 trial 编号应从 1 开始，而不是 Optuna 内部的 0。"""
    config = {
        "model": {"hidden_dim": 256, "branch_dim": 128},
        "training": {"lr": 0.0018259913257430906, "kl_beta": 1.7099712671569545e-05, "epochs": 150, "patience": 15},
    }

    text = format_trial_config(0, config)

    assert text == (
        "\n========== Optuna Trial 1 ==========\n"
        "Sampled params:\n"
        "  hidden_dim: 256\n"
        "  branch_dim: 128\n"
        "  lr: 0.0018259913257430906\n"
        "  kl_beta: 1.7099712671569545e-05\n"
        "Training options:\n"
        "  epochs: 150\n"
        "  patience: 15\n"
        "====================================="
    )


def test_format_trial_config_can_color_trial_number_and_params():
    """彩色控制台输出应只突出 trial 编号和参数区域。"""
    signature = inspect.signature(format_trial_config)
    assert "color" in signature.parameters, "format_trial_config should accept a color flag"

    config = {
        "model": {"hidden_dim": 256, "branch_dim": 128},
        "training": {"lr": 0.0018259913257430906, "kl_beta": 1.7099712671569545e-05, "epochs": 150, "patience": 15},
    }

    text = format_trial_config(0, config, color=True)

    assert "\033[32mOptuna Trial 1\033[0m" in text
    assert "\033[34mSampled params:\033[0m" in text
    assert "\033[34m  hidden_dim: 256\033[0m" in text
    assert "\033[34m  kl_beta: 1.7099712671569545e-05\033[0m" in text
    assert "\033[31m" not in text


def test_format_duration_uses_hh_mm_ss():
    """调参和训练摘要使用同一种耗时格式。"""
    assert format_duration(3723.4) == "01:02:03"


def test_format_trial_result_prints_duration_objective_and_metrics(tmp_path):
    """单个 trial 结束后应格式化输出用时、目标指标和验证集指标。"""
    metrics = {
        "rmse": 10.123456,
        "mae": 5.5,
        "crps": 0.25,
        "picp_90": 0.91,
        "nll": 7.123456,
    }

    text = format_trial_result(
        trial_number=0,
        run_dir=tmp_path / "trial-run",
        metrics=metrics,
        objective_metric="crps",
        elapsed_seconds=65.4,
    )

    assert text == (
        "\n========== Optuna Trial 1 Complete ==========\n"
        "elapsed_time: 00:01:05\n"
        "objective_metric: crps\n"
        "objective_value: 0.250000\n"
        "Validation metrics:\n"
        "  rmse: 10.123456\n"
        "  mae: 5.500000\n"
        "  crps: 0.250000\n"
        "  picp_90: 0.910000\n"
        "  nll: 7.123456\n"
        f"run_dir: {tmp_path / 'trial-run'}\n"
        "=============================================="
    )


def test_format_trial_result_can_color_trial_number_and_metrics(tmp_path):
    """Trial 结果摘要应让指标区域用紫色，但不使用红色。"""
    signature = inspect.signature(format_trial_result)
    assert "color" in signature.parameters, "format_trial_result should accept a color flag"
    metrics = {"rmse": 10.123456, "crps": 0.25}

    text = format_trial_result(
        trial_number=0,
        run_dir=tmp_path / "trial-run",
        metrics=metrics,
        objective_metric="crps",
        elapsed_seconds=65.4,
        color=True,
    )

    assert "\033[32mOptuna Trial 1\033[0m" in text
    assert "\033[35mobjective_metric: crps\033[0m" in text
    assert "\033[35m  rmse: 10.123456\033[0m" in text
    assert "\033[35m  crps: 0.250000\033[0m" in text
    assert "\033[31m" not in text


def test_format_study_summary_prints_best_params_on_multiple_lines(tmp_path):
    """调参结束摘要应展示目标指标、总用时和 best params。"""

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

    text = format_study_summary(
        FakeStudy(),
        tmp_path / "tuning" / "fixed",
        objective_metric="crps",
        elapsed_seconds=3723.4,
    )

    assert text == (
        "\nTuning Summary\n"
        "elapsed_time: 01:02:03\n"
        "objective_metric: crps\n"
        "\nBest trial:\n"
        "  number: 11\n"
        "  value: 2448.0612226049443\n"
        "\nBest params:\n"
        "  hidden_dim: 256\n"
        "  branch_dim: 64\n"
        "  lr: 0.00019544293665253097\n"
        "  kl_beta: 1.2114960587728605e-05\n"
        "\nTuning results exported to:\n"
        f"  {tmp_path / 'tuning' / 'fixed'}"
    )


def test_format_optuna_trial_log_prints_best_trial_and_params_with_color():
    """替代 Optuna 默认单行日志的摘要应分区着色。"""
    assert hasattr(tune, "format_optuna_trial_log"), "missing formatted Optuna trial log helper"

    class FakeFrozenTrial:
        number = 0
        value = 600.5989660198786
        params = {
            "hidden_dim": 128,
            "branch_dim": 256,
            "lr": 0.0022497849788863807,
            "kl_beta": 0.00014479268325065474,
        }

    class FakeStudy:
        best_value = 600.5989660198786

        class BestTrial:
            number = 0

        best_trial = BestTrial()

    text = tune.format_optuna_trial_log(FakeStudy(), FakeFrozenTrial(), color=True)

    assert "\033[32mOptuna Trial 1\033[0m" in text
    assert "\033[35mvalue: 600.5989660198786\033[0m" in text
    assert "\033[32m  number: 1\033[0m" in text
    assert "\033[34mParams:\033[0m" in text
    assert "\033[34m  lr: 0.0022497849788863807\033[0m" in text
    assert "\033[31m" not in text


def test_configure_optuna_logging_suppresses_default_info_logs():
    """Optuna 自带的一行 INFO 日志应关闭，避免和格式化输出重复。"""
    assert hasattr(tune, "configure_optuna_logging"), "missing Optuna logging configuration helper"

    calls = []

    class FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(level):
            calls.append(("set_verbosity", level))

        @staticmethod
        def disable_default_handler():
            calls.append(("disable_default_handler",))

    class FakeOptuna:
        logging = FakeLogging

    tune.configure_optuna_logging(FakeOptuna)

    assert calls == [("set_verbosity", "WARNING"), ("disable_default_handler",)]


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
        "tuning": {"objective_metric": "crps"},
    }

    export_dir = export_study_results(FakeStudy(), base_config, run_name="fixed")

    assert export_dir == tmp_path / "tuning" / "fixed"
    summary = json.loads((export_dir / "best_params.json").read_text(encoding="utf-8"))
    assert summary["objective_metric"] == "crps"
    assert summary["best_trial"] == 10
    assert summary["best_value"] == 2448.0612226049443
    assert summary["best_params"]["hidden_dim"] == 256
    assert (export_dir / "trials.csv").exists()
    best_config = yaml.safe_load((export_dir / "best_config.yaml").read_text(encoding="utf-8"))
    assert best_config["model"]["hidden_dim"] == 256
    assert best_config["training"]["kl_beta"] == 1.2114960587728605e-05
