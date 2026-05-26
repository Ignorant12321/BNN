from __future__ import annotations

from pathlib import Path

import yaml

import src.experiments.tune as tune_module
from src.experiments.tune import build_trial_config, read_metric_from_run, run_tuning
from src.artifacts.run_io import create_run_dir


class DummyTrial:
    number = 0

    def suggest_float(self, name, low, high, log=False):
        return high if log else low

    def suggest_int(self, name, low, high, log=False):
        return high

    def suggest_categorical(self, name, choices):
        return choices[-1]

    def set_user_attr(self, name, value):
        pass


def test_build_trial_config_applies_search_space_to_nested_config():
    base_config = {
        "model": {"name": "improved_bnn", "hidden_dim": 128},
        "training": {"lr": 0.0005, "weight_decay": 0.0001},
    }
    search_space = {
        "lr": {"type": "log_float", "low": 0.0001, "high": 0.001},
        "model.branch_dim": {"type": "categorical", "choices": [32, 64]},
        "training.batch_size": {"type": "int", "low": 16, "high": 64},
    }

    trial_config, params = build_trial_config(base_config, search_space, DummyTrial())

    assert trial_config["training"]["lr"] == 0.001
    assert trial_config["model"]["branch_dim"] == 64
    assert trial_config["training"]["batch_size"] == 64
    assert params == {"lr": 0.001, "model.branch_dim": 64, "training.batch_size": 64}


def test_read_metric_from_run_reads_metrics_csv(tmp_path: Path):
    (tmp_path / "metrics.csv").write_text(
        "split,metric,value\ntrain,rmse,3.0\nval,rmse,1.25\n",
        encoding="utf-8",
    )

    assert read_metric_from_run(tmp_path, "val_rmse") == 1.25


def test_read_metric_from_run_supports_split_names_with_underscores(tmp_path: Path):
    (tmp_path / "metrics.csv").write_text(
        "split,metric,value\nval_generation,nrmse,0.052\nval,rmse,1.25\n",
        encoding="utf-8",
    )

    assert read_metric_from_run(tmp_path, "val_generation_nrmse") == 0.052


def test_run_tuning_uses_sqlite_storage_and_resumes_to_target_trials(tmp_path: Path, monkeypatch):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "outputs"),
                "data": {
                    "lookback": 1,
                    "horizon": 1,
                    "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
                },
                "model": {"name": "improved_bnn"},
                "training": {"backend": "numpy", "epochs": 1},
            }
        ),
        encoding="utf-8",
    )
    tune_config_path = tmp_path / "tune.yaml"
    tune_config = {
        "name": "unit_bnn",
        "base_config": str(base_config_path),
        "output_dir": str(tmp_path / "outputs"),
        "n_trials": 1,
        "metric": "val_rmse",
        "direction": "minimize",
        "search_space": {"lr": {"type": "categorical", "choices": [0.001, 0.0005]}},
    }
    tune_config_path.write_text(yaml.safe_dump(tune_config), encoding="utf-8")
    calls = []

    def fake_run_training(config):
        calls.append(config)
        run_dir = Path(config["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.csv").write_text(
            f"split,metric,value\nval,rmse,{10 - len(calls)}\n",
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(tune_module, "run_training", fake_run_training)

    first = run_tuning(tune_config_path, note="unit tuning search")
    assert len(calls) == 1
    assert first["completed_trials"] == 1
    assert Path(calls[0]["run_dir"]).parent == tmp_path / "outputs" / "tuning" / "unit_bnn" / "runs"
    assert (tmp_path / "outputs" / "tuning" / "unit_bnn" / "note.txt").read_text(encoding="utf-8") == "unit tuning search\n"

    tune_config["n_trials"] = 2
    tune_config_path.write_text(yaml.safe_dump(tune_config), encoding="utf-8")
    second = run_tuning(tune_config_path)
    assert len(calls) == 2
    assert second["completed_trials"] == 2

    third = run_tuning(tune_config_path)
    assert len(calls) == 2
    assert third["completed_trials"] == 2
    assert (tmp_path / "outputs" / "tuning" / "unit_bnn" / "study.db").is_file()
    assert (tmp_path / "outputs" / "tuning" / "unit_bnn" / "trials.csv").is_file()
    assert (tmp_path / "outputs" / "tuning" / "unit_bnn" / "best_config.yaml").is_file()
    assert (tmp_path / "outputs" / "tuning" / "unit_bnn" / "best_run.txt").is_file()


def test_run_tuning_writes_default_note_from_tuning_dir_name(tmp_path: Path, monkeypatch):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "outputs"),
                "data": {
                    "lookback": 1,
                    "horizon": 1,
                    "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
                },
                "model": {"name": "improved_bnn"},
                "training": {"backend": "numpy", "epochs": 1},
            }
        ),
        encoding="utf-8",
    )
    tune_config_path = tmp_path / "tune.yaml"
    tune_config_path.write_text(
        yaml.safe_dump(
            {
                "name": "default_note",
                "base_config": str(base_config_path),
                "output_dir": str(tmp_path / "outputs"),
                "n_trials": 0,
                "metric": "val_rmse",
                "direction": "minimize",
                "search_space": {},
            }
        ),
        encoding="utf-8",
    )

    run_tuning(tune_config_path)

    assert (tmp_path / "outputs" / "tuning" / "default_note" / "note.txt").read_text(encoding="utf-8") == "default_note\n"


def test_create_run_dir_honors_explicit_run_dir(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "tuning" / "study" / "runs" / "trial-0000"

    created = create_run_dir({"model": {"name": "improved_bnn"}, "run_dir": str(run_dir)})

    assert created == run_dir
    assert created.is_dir()


def test_bnn_tuning_configs_use_fixed_batch_size_and_distinct_4h_study():
    root = Path(__file__).resolve().parents[1]
    main_config = yaml.safe_load((root / "configs" / "tune" / "bnn.yaml").read_text(encoding="utf-8"))
    four_hour_config = yaml.safe_load((root / "configs" / "tune" / "bnn_4h.yaml").read_text(encoding="utf-8"))
    pv_usibnn_config = yaml.safe_load((root / "configs" / "tune" / "pv_usibnn.yaml").read_text(encoding="utf-8"))

    assert main_config["search_space"].get("batch_size") is None
    assert four_hour_config["search_space"].get("batch_size") is None
    assert pv_usibnn_config["search_space"].get("batch_size") is None
    assert four_hour_config["base_config"] == "../models/bnn/4h.yaml"
    assert four_hour_config["name"] == "bnn_4h_generation_optuna"
    assert four_hour_config["study_name"] == "bnn_4h_generation_optuna"
    assert four_hour_config["metric"] == "val_generation_nrmse"
    assert pv_usibnn_config["base_config"] == "../models/bnn/pv_usibnn.yaml"
    assert pv_usibnn_config["name"] == "pv_usibnn_generation_optuna"
    assert pv_usibnn_config["study_name"] == "pv_usibnn_generation_optuna"
    assert pv_usibnn_config["metric"] == "val_generation_nrmse"
