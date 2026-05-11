"""Tests for applying Optuna best parameters to training configs."""

from __future__ import annotations

import json
import subprocess
import sys

import yaml

from src.apply_tuning import apply_best_params, find_latest_best_params, format_change, ParamChange, resolve_default_objective


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


def test_apply_best_params_preview_reports_without_writing(tmp_path):
    target_path = tmp_path / "default.yaml"
    target_path.write_text(yaml.safe_dump({"training": {"kl_beta": 0.0001}}, sort_keys=False), encoding="utf-8")
    source_path = tmp_path / "best_params.json"
    source_path.write_text(json.dumps({"best_params": {"kl_beta": 0.00001}}), encoding="utf-8")

    changes = apply_best_params(source_path, target_path, write=False)

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


def test_find_latest_best_params_uses_newest_directory_when_objective_matches(tmp_path):
    rmse_dir = tmp_path / "20260510-100000"
    crps_dir = tmp_path / "20260510-110000"
    rmse_dir.mkdir()
    crps_dir.mkdir()
    (rmse_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "rmse", "best_params": {"lr": 0.001}}),
        encoding="utf-8",
    )
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )

    assert find_latest_best_params(tmp_path, objective_metric="crps") == crps_dir / "best_params.json"


def test_find_latest_best_params_rejects_objective_mismatch_in_newest_directory(tmp_path):
    rmse_dir = tmp_path / "20260510-100000"
    crps_dir = tmp_path / "20260510-110000"
    rmse_dir.mkdir()
    crps_dir.mkdir()
    (rmse_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "rmse", "best_params": {"lr": 0.001}}),
        encoding="utf-8",
    )
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )

    try:
        find_latest_best_params(tmp_path, objective_metric="rmse")
    except FileNotFoundError as exc:
        assert "Latest best_params.json" in str(exc)
        assert "objective_metric='crps'" in str(exc)
    else:
        raise AssertionError("expected objective mismatch to fail")


def test_find_latest_best_params_treats_latest_legacy_summary_as_rmse(tmp_path):
    legacy_dir = tmp_path / "20260510-100000"
    crps_dir = tmp_path / "20260510-110000"
    newest_legacy_dir = tmp_path / "20260510-120000"
    legacy_dir.mkdir()
    crps_dir.mkdir()
    newest_legacy_dir.mkdir()
    (legacy_dir / "best_params.json").write_text(json.dumps({"best_params": {"lr": 0.001}}), encoding="utf-8")
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )
    (newest_legacy_dir / "best_params.json").write_text(json.dumps({"best_params": {"lr": 0.003}}), encoding="utf-8")

    assert find_latest_best_params(tmp_path, objective_metric="rmse") == newest_legacy_dir / "best_params.json"


def test_resolve_default_objective_reads_tuning_config(tmp_path):
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(yaml.safe_dump({"tuning": {"objective_metric": "crps"}}), encoding="utf-8")

    assert resolve_default_objective(config_path) == "crps"


def test_format_change_can_color_path_and_status():
    changed = format_change(ParamChange("training.lr", 0.001, 0.002), color=True)
    unchanged = format_change(ParamChange("model.hidden_dim", 256, 256), color=True)

    assert "\033[34mtraining.lr\033[0m" in changed
    assert "\033[32m(changed)\033[0m" in changed
    assert "\033[34mmodel.hidden_dim\033[0m" in unchanged
    assert "\033[35m(unchanged)\033[0m" in unchanged


def test_apply_tuning_cli_previews_and_waits_for_confirmation(tmp_path):
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
            "--no-color",
        ],
        input="n\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "model.hidden_dim: 128 -> 128 (unchanged)" in result.stdout
    assert "training.lr: 0.001 -> 0.0002 (changed)" in result.stdout
    assert "Apply these changes? [y/N]" in result.stdout
    assert "No changes written." in result.stdout
    assert yaml.safe_load(target_path.read_text(encoding="utf-8"))["training"]["lr"] == 0.001


def test_apply_tuning_cli_yes_updates_target_without_prompt(tmp_path):
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
            "--yes",
            "--no-color",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Apply these changes? [y/N]" not in result.stdout
    assert "Updated target config" in result.stdout
    assert yaml.safe_load(target_path.read_text(encoding="utf-8"))["training"]["lr"] == 0.0002


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
            "--yes",
            "--no-color",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Source: {crps_dir / 'best_params.json'}" in result.stdout
    assert "Objective: crps" in result.stdout
    assert yaml.safe_load(target_path.read_text(encoding="utf-8"))["training"]["lr"] == 0.002


def test_apply_tuning_cli_reports_missing_objective_without_traceback(tmp_path):
    tuning_dir = tmp_path / "tuning"
    crps_dir = tuning_dir / "20260510-110000"
    crps_dir.mkdir(parents=True)
    (crps_dir / "best_params.json").write_text(
        json.dumps({"objective_metric": "crps", "best_params": {"lr": 0.002}}),
        encoding="utf-8",
    )
    target_path = tmp_path / "default.yaml"
    target_path.write_text(yaml.safe_dump({"training": {"lr": 0.01}}, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.apply_tuning",
            "--tuning-dir",
            str(tuning_dir),
            "--objective",
            "rmse",
            "--target",
            str(target_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Latest best_params.json" in result.stderr
    assert "objective_metric='crps'" in result.stderr
    assert "not objective_metric='rmse'" in result.stderr
    assert "python -m src.select_tuning" in result.stderr
    assert "Traceback" not in result.stderr
