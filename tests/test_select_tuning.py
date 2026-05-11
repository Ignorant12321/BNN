"""Tests for selecting existing tuning trials by validation metrics."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

import yaml

from src.select_tuning import ArtifactChange, find_latest_tuning_run, format_selection, select_best_trial


def write_tuning_run(base):
    run_dir = base / "20260511-025155"
    model_dir = run_dir / "improved_bnn"
    model_dir.mkdir(parents=True)
    rows = [
        {
            "number": "0",
            "params_hidden_dim": "128",
            "params_branch_dim": "64",
            "params_lr": "0.001",
            "params_kl_beta": "0.0001",
            "state": "COMPLETE",
        },
        {
            "number": "1",
            "params_hidden_dim": "256",
            "params_branch_dim": "128",
            "params_lr": "0.002",
            "params_kl_beta": "0.00001",
            "state": "COMPLETE",
        },
    ]
    with open(run_dir / "trials.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "best_params.json").write_text(
        json.dumps(
            {
                "objective_metric": "crps",
                "best_trial": 0,
                "best_value": 3.0,
                "best_params": {
                    "hidden_dim": 128,
                    "branch_dim": 64,
                    "lr": 0.001,
                    "kl_beta": 0.0001,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"name": "improved_bnn", "hidden_dim": 128, "branch_dim": 64},
                "training": {"lr": 0.001, "kl_beta": 0.0001, "epochs": 10},
                "tuning": {"objective_metric": "crps"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    first = model_dir / "20260511-025157" / "metrics"
    second = model_dir / "20260511-025330" / "metrics"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "validation_metrics.json").write_text(
        json.dumps({"rmse": 10.0, "mae": 4.0, "crps": 3.0, "nll": 2.0}),
        encoding="utf-8",
    )
    (second / "validation_metrics.json").write_text(
        json.dumps({"rmse": 8.0, "mae": 5.0, "crps": 3.5, "nll": 2.5}),
        encoding="utf-8",
    )
    return run_dir


def test_select_best_trial_uses_requested_validation_metric(tmp_path):
    run_dir = write_tuning_run(tmp_path)

    selected = select_best_trial(run_dir, "rmse")

    assert selected.number == 1
    assert selected.metric == "rmse"
    assert selected.metric_value == 8.0
    assert selected.params == {
        "hidden_dim": 256,
        "branch_dim": 128,
        "lr": 0.002,
        "kl_beta": 0.00001,
    }


def test_find_latest_tuning_run_uses_newest_directory_with_trials_csv(tmp_path):
    old_dir = write_tuning_run(tmp_path)
    new_dir = write_tuning_run(tmp_path / "nested")

    assert find_latest_tuning_run(tmp_path) == old_dir
    assert find_latest_tuning_run(tmp_path / "nested") == new_dir


def test_format_selection_can_color_headings_metrics_and_change_status(tmp_path):
    selected = select_best_trial(write_tuning_run(tmp_path), "rmse")
    text = format_selection(
        selected,
        [ArtifactChange("best_params.json best_trial", 0, 1)],
        color=True,
    )

    assert "\033[32mSelected trial:\033[0m 1" in text
    assert "\033[35mmetric:\033[0m rmse=8.000000" in text
    assert "  rmse: 8.000000" in text
    assert "\033[34mArtifact changes:\033[0m" in text
    assert "\033[32m(changed)\033[0m" in text


def test_select_tuning_cli_previews_and_waits_for_confirmation(tmp_path):
    tuning_dir = tmp_path / "tuning"
    run_dir = write_tuning_run(tuning_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.select_tuning",
            "--tuning-dir",
            str(tuning_dir),
            "--metric",
            "rmse",
            "--no-color",
        ],
        input="n\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Selected trial: 1" in result.stdout
    assert "metric: rmse=8.000000" in result.stdout
    assert "best_params.json objective_metric: crps -> rmse (changed)" in result.stdout
    assert "best_params.json best_trial: 0 -> 1 (changed)" in result.stdout
    assert "Apply these changes? [y/N]" in result.stdout
    assert "No changes written." in result.stdout
    summary = json.loads((run_dir / "best_params.json").read_text(encoding="utf-8"))
    assert summary["objective_metric"] == "crps"
    assert summary["best_trial"] == 0


def test_select_tuning_cli_yes_updates_artifacts_without_prompt(tmp_path):
    tuning_dir = tmp_path / "tuning"
    run_dir = write_tuning_run(tuning_dir)

    confirmed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.select_tuning",
            "--tuning-dir",
            str(tuning_dir),
            "--metric",
            "rmse",
            "--yes",
            "--no-color",
        ],
        capture_output=True,
        text=True,
    )

    assert confirmed.returncode == 0, confirmed.stderr
    assert "Apply these changes? [y/N]" not in confirmed.stdout
    assert "Updated tuning artifacts" in confirmed.stdout
    summary = json.loads((run_dir / "best_params.json").read_text(encoding="utf-8"))
    assert summary["objective_metric"] == "rmse"
    assert summary["best_trial"] == 1
    assert summary["best_value"] == 8.0
    assert summary["best_params"]["hidden_dim"] == 256
    best_config = yaml.safe_load((run_dir / "best_config.yaml").read_text(encoding="utf-8"))
    assert best_config["model"]["hidden_dim"] == 256
    assert best_config["training"]["lr"] == 0.002
    assert best_config["tuning"]["objective_metric"] == "rmse"


def test_select_tuning_cli_without_stdin_defaults_to_no_changes(tmp_path):
    tuning_dir = tmp_path / "tuning"
    run_dir = write_tuning_run(tuning_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.select_tuning",
            "--tuning-dir",
            str(tuning_dir),
            "--metric",
            "rmse",
            "--no-color",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "No changes written." in result.stdout
    assert json.loads((run_dir / "best_params.json").read_text(encoding="utf-8"))["best_trial"] == 0


def test_apply_tuning_can_apply_metric_selected_by_select_tuning(tmp_path):
    tuning_dir = tmp_path / "tuning"
    write_tuning_run(tuning_dir)
    target_path = tmp_path / "default.yaml"
    target_path.write_text(
        yaml.safe_dump({"model": {"hidden_dim": 128}, "training": {"lr": 0.001}}, sort_keys=False),
        encoding="utf-8",
    )

    select_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.select_tuning",
            "--tuning-dir",
            str(tuning_dir),
            "--metric",
            "rmse",
            "--yes",
            "--no-color",
        ],
        capture_output=True,
        text=True,
    )
    assert select_result.returncode == 0, select_result.stderr

    apply_result = subprocess.run(
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
            "--yes",
            "--no-color",
        ],
        capture_output=True,
        text=True,
    )

    assert apply_result.returncode == 0, apply_result.stderr
    assert "Objective: rmse" in apply_result.stdout
    updated = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert updated["model"]["hidden_dim"] == 256
    assert updated["training"]["lr"] == 0.002
