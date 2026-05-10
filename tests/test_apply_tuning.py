"""Tests for applying Optuna best parameters to training configs."""

from __future__ import annotations

import json
import subprocess
import sys

import yaml

from src.apply_tuning import apply_best_params, find_latest_best_params, resolve_default_objective


def test_apply_best_params_updates_only_optuna_fields_and_reports_changes(tmp_path):
    target_path = tmp_path / "default.yaml"
    target_path.write_text(
        yaml.safe_dump(
            {
                "model": {"hidden_dim": 128, "branch_dim": 64, "prior_sigma": 1.0},
                "training": {"lr": 0.001, "kl_beta": 0.0001, "epochs": 350},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source_path = tmp_path / "best_params.json"
    source_path.write_text(
        json.dumps(
            {
                "best_trial": 7,
                "best_value": 123.4,
                "best_params": {
                    "hidden_dim": 256,
                    "branch_dim": 64,
                    "lr": 0.0002,
                    "unknown": "ignored",
                },
            }
        ),
        encoding="utf-8",
    )

    changes = apply_best_params(source_path, target_path)

    updated = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert updated["model"] == {"hidden_dim": 256, "branch_dim": 64, "prior_sigma": 1.0}
    assert updated["training"] == {"lr": 0.0002, "kl_beta": 0.0001, "epochs": 350}
    assert [change.as_row() for change in changes] == [
        {"path": "model.hidden_dim", "old": 128, "new": 256, "status": "changed"},
        {"path": "model.branch_dim", "old": 64, "new": 64, "status": "unchanged"},
        {"path": "training.lr", "old": 0.001, "new": 0.0002, "status": "changed"},
    ]


def test_apply_best_params_dry_run_reports_without_writing(tmp_path):
    target_path = tmp_path / "default.yaml"
    target_path.write_text(yaml.safe_dump({"training": {"kl_beta": 0.0001}}, sort_keys=False), encoding="utf-8")
    source_path = tmp_path / "best_params.json"
    source_path.write_text(json.dumps({"best_params": {"kl_beta": 0.00001}}), encoding="utf-8")

    changes = apply_best_params(source_path, target_path, dry_run=True)

    assert yaml.safe_load(target_path.read_text(encoding="utf-8"))["training"]["kl_beta"] == 0.0001
    assert [change.as_row() for change in changes] == [
        {"path": "training.kl_beta", "old": 0.0001, "new": 0.00001, "status": "changed"}
    ]


def test_find_latest_best_params_uses_newest_tuning_directory(tmp_path):
    old_dir = tmp_path / "20260509-100000"
    new_dir = tmp_path / "20260510-100000"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "best_params.json").write_text(json.dumps({"best_params": {"lr": 0.001}}), encoding="utf-8")
    (new_dir / "best_params.json").write_text(json.dumps({"best_params": {"lr": 0.002}}), encoding="utf-8")

    assert find_latest_best_params(tmp_path) == new_dir / "best_params.json"


def test_find_latest_best_params_filters_by_objective_metric(tmp_path):
    rmse_dir = tmp_path / "20260510-100000"
    crps_dir = tmp_path / "20260510-110000"
    newest_rmse_dir = tmp_path / "20260510-120000"
    rmse_dir.mkdir()
    crps_dir.mkdir()
    newest_rmse_dir.mkdir()
    (rmse_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "rmse", "best_params": {"lr": 0.001}}),
        encoding="utf-8",
    )
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )
    (newest_rmse_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "rmse", "best_params": {"lr": 0.003}}),
        encoding="utf-8",
    )

    assert find_latest_best_params(tmp_path, objective_metric="crps") == crps_dir / "best_params.json"


def test_find_latest_best_params_treats_legacy_summaries_as_rmse(tmp_path):
    legacy_dir = tmp_path / "20260510-100000"
    crps_dir = tmp_path / "20260510-110000"
    legacy_dir.mkdir()
    crps_dir.mkdir()
    (legacy_dir / "best_params.json").write_text(json.dumps({"best_params": {"lr": 0.001}}), encoding="utf-8")
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )

    assert find_latest_best_params(tmp_path, objective_metric="rmse") == legacy_dir / "best_params.json"


def test_resolve_default_objective_reads_tuning_config(tmp_path):
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(yaml.safe_dump({"tuning": {"objective_metric": "crps"}}), encoding="utf-8")

    assert resolve_default_objective(config_path) == "crps"


def test_apply_tuning_cli_prints_changed_and_unchanged_fields(tmp_path):
    target_path = tmp_path / "default.yaml"
    target_path.write_text(
        yaml.safe_dump({"model": {"hidden_dim": 128}, "training": {"lr": 0.001}}, sort_keys=False),
        encoding="utf-8",
    )
    source_path = tmp_path / "best_params.json"
    source_path.write_text(json.dumps({"best_params": {"hidden_dim": 128, "lr": 0.0002}}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.apply_tuning",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "model.hidden_dim: 128 -> 128 (unchanged)" in result.stdout
    assert "training.lr: 0.001 -> 0.0002 (changed)" in result.stdout


def test_apply_tuning_cli_uses_config_objective_for_latest_source(tmp_path):
    tuning_dir = tmp_path / "tuning"
    rmse_dir = tuning_dir / "20260510-100000"
    crps_dir = tuning_dir / "20260510-110000"
    rmse_dir.mkdir(parents=True)
    crps_dir.mkdir()
    (rmse_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "rmse", "best_params": {"lr": 0.001}}),
        encoding="utf-8",
    )
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(yaml.safe_dump({"tuning": {"objective_metric": "crps"}}), encoding="utf-8")
    target_path = tmp_path / "default.yaml"
    target_path.write_text(yaml.safe_dump({"training": {"lr": 0.01}}, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.apply_tuning",
            "--tuning-dir",
            str(tuning_dir),
            "--config",
            str(config_path),
            "--target",
            str(target_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Source: {crps_dir / 'best_params.json'}" in result.stdout
    assert "Objective: crps" in result.stdout
    assert yaml.safe_load(target_path.read_text(encoding="utf-8"))["training"]["lr"] == 0.002
