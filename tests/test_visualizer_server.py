from pathlib import Path
import subprocess
import sys

import pytest

from visualizer.server import (
    collect_run,
    create_comparison,
    discover_runs,
    ensure_project_root_on_syspath,
    list_comparisons,
    list_train_runs,
    read_comparison,
)
from src.experiments.train import run_training


def _write_small_processed_splits(processed_dir: Path) -> None:
    processed_dir.mkdir()
    frame = "\n".join(
        [
            "DATE_TIME,AC_POWER,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION",
            "2020-01-01 07:00:00,1.0,20.0,25.0,0.1",
            "2020-01-01 07:15:00,2.0,21.0,26.0,0.2",
            "2020-01-01 07:30:00,3.0,22.0,27.0,0.3",
            "2020-01-01 07:45:00,4.0,23.0,28.0,0.4",
            "2020-01-01 08:00:00,5.0,24.0,29.0,0.5",
            "2020-01-01 08:15:00,6.0,25.0,30.0,0.6",
        ]
    )
    for split_name in ("train", "val", "test"):
        (processed_dir / f"{split_name}.csv").write_text(frame + "\n", encoding="utf-8")


def _zero_hour_bnn_config(tmp_path: Path, processed_dir: Path) -> dict:
    return {
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 0,
            "horizon": 1,
            "features": {
                "history": [],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "improved_bnn"},
        "training": {
            "backend": "torch",
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "kl_beta": 0.0,
            "weight_decay": 0.0,
            "early_stopping": {"enabled": False},
        },
        "evaluation": {"n_samples": 2},
    }


def test_discover_runs_reads_training_artifacts(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "train" / "improved_bnn" / "20260521-170306"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  name: improved_bnn\ndata:\n  lookback: 4\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text("split,metric,value\nval,rmse,1.5\ntrain,mae,2.0\n", encoding="utf-8")
    (run_dir / "note.txt").write_text("one hour\n", encoding="utf-8")
    (figures_dir / "loss_curve.png").write_bytes(b"\x89PNG\r\n")

    runs = discover_runs(tmp_path / "outputs")

    assert [run["label"] for run in runs] == ["20260521-170306"]
    assert runs[0]["model"] == "improved_bnn"
    assert runs[0]["note"] == "one hour"
    assert runs[0]["metrics"]["val_rmse"] == 1.5
    assert runs[0]["config"]["data"]["lookback"] == 4
    assert runs[0]["figures"][0]["name"] == "loss_curve.png"


def test_collect_run_uses_relative_paths_for_browser_links(tmp_path: Path):
    project_root = tmp_path
    run_dir = project_root / "outputs" / "train" / "mlp_baseline" / "20260521-170306"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("split,metric,value\ntest,rmse,3.25\n", encoding="utf-8")

    run = collect_run(run_dir, project_root)

    assert run["path"] == "outputs/train/mlp_baseline/20260521-170306"
    assert run["label"] == "20260521-170306"
    assert run["metrics"]["test_rmse"] == 3.25


def test_discover_runs_reads_compare_model_metrics(tmp_path: Path):
    source_run = tmp_path / "outputs" / "train" / "improved_bnn" / "20260521-170306"
    source_run.mkdir(parents=True)
    (source_run / "config.yaml").write_text("model:\n  name: improved_bnn\n", encoding="utf-8")
    compare_dir = tmp_path / "outputs" / "comparisons" / "20260521-171302"
    compare_dir.mkdir(parents=True)
    (compare_dir / "model_metrics.csv").write_text(
        "label,model,run_dir,test_mae,test_rmse\n"
        "BNN-1h,improved_bnn,outputs/train/improved_bnn/20260521-170306,10.0,20.0\n",
        encoding="utf-8",
    )

    runs = discover_runs(tmp_path / "outputs")

    comparison_run = next(run for run in runs if run["label"] == "BNN-1h")
    assert comparison_run["path"] == "outputs/comparisons/20260521-171302#BNN-1h"
    assert comparison_run["metrics"]["test_rmse"] == 20.0
    assert comparison_run["config"]["model"]["name"] == "improved_bnn"


def test_list_comparisons_returns_timestamp_folders_with_run_counts(tmp_path: Path):
    old_dir = tmp_path / "outputs" / "comparisons" / "20260521-171302"
    new_dir = tmp_path / "outputs" / "comparisons" / "20260521-174633"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "model_metrics.csv").write_text("label,model,run_dir,test_rmse\nA,m,path,1.0\n", encoding="utf-8")
    (new_dir / "model_metrics.csv").write_text(
        "label,model,run_dir,test_rmse\nA,m,path,1.0\nB,m,path,2.0\n",
        encoding="utf-8",
    )
    (new_dir / "note.txt").write_text("lookback comparison\n", encoding="utf-8")

    comparisons = list_comparisons(tmp_path)

    assert comparisons == [
        {
            "name": "20260521-174633",
            "path": "outputs/comparisons/20260521-174633",
            "note": "lookback comparison",
            "runCount": 2,
        },
        {
            "name": "20260521-171302",
            "path": "outputs/comparisons/20260521-171302",
            "note": "",
            "runCount": 1,
        },
    ]


def test_read_comparison_reads_only_selected_timestamp_folder(tmp_path: Path):
    source_run = tmp_path / "outputs" / "train" / "improved_bnn" / "20260521-170306"
    source_run.mkdir(parents=True)
    (source_run / "config.yaml").write_text("model:\n  name: improved_bnn\n", encoding="utf-8")
    first_dir = tmp_path / "outputs" / "comparisons" / "20260521-171302"
    second_dir = tmp_path / "outputs" / "comparisons" / "20260521-174633"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    (first_dir / "model_metrics.csv").write_text(
        "label,model,run_dir,test_rmse\n"
        "first,improved_bnn,outputs/train/improved_bnn/20260521-170306,1.0\n",
        encoding="utf-8",
    )
    (second_dir / "model_metrics.csv").write_text(
        "label,model,run_dir,test_rmse\n"
        "second,improved_bnn,outputs/train/improved_bnn/20260521-170306,2.0\n",
        encoding="utf-8",
    )

    payload = read_comparison("outputs/comparisons/20260521-174633", tmp_path)

    assert payload["comparison"]["name"] == "20260521-174633"
    assert [run["label"] for run in payload["runs"]] == ["second"]
    assert payload["runs"][0]["metrics"]["test_rmse"] == 2.0


def test_read_comparison_includes_comparison_figures_and_prediction_summaries(tmp_path: Path):
    compare_dir = tmp_path / "outputs" / "comparisons" / "20260521-174633"
    figures_dir = compare_dir / "figures"
    predictions_dir = compare_dir / "predictions"
    figures_dir.mkdir(parents=True)
    predictions_dir.mkdir()
    (compare_dir / "model_metrics.csv").write_text(
        "label,model,run_dir,test_mae,test_picp_90\n"
        "BNN-1h,improved_bnn,outputs/train/improved_bnn/20260521-170306,10.0,0.91\n",
        encoding="utf-8",
    )
    (figures_dir / "metrics_test_mae.png").write_bytes(b"\x89PNG\r\n")
    (figures_dir / "loss_curves.png").write_bytes(b"\x89PNG\r\n")
    (figures_dir / "prediction_0800_1200.png").write_bytes(b"\x89PNG\r\n")
    (predictions_dir / "BNN-1h.csv").write_text(
        "label,sample,horizon,target,mean,log_var,std,lower_90,upper_90,lower_95,upper_95,target_time\n"
        "BNN-1h,0,0,1.0,1.5,0.0,1.0,0.0,2.0,-0.5,2.5,2020-01-01 00:00:00\n"
        "BNN-1h,0,1,2.0,1.0,0.0,1.0,0.5,1.5,0.0,2.0,2020-01-01 00:15:00\n",
        encoding="utf-8",
    )

    payload = read_comparison("20260521-174633", tmp_path)

    assert [(figure["group"], figure["name"]) for figure in payload["figures"]] == [
        ("loss", "loss_curves.png"),
        ("predict", "prediction_0800_1200.png"),
    ]
    assert payload["runs"][0]["predictionSummary"] == {
        "rows": 2,
        "horizons": 2,
        "targetMean": 1.5,
        "predictionMean": 1.25,
        "mae": 0.75,
        "picp90": 0.5,
        "pinaw90": 1.5,
        "picp95": 1.0,
        "pinaw95": 2.5,
    }


def test_read_comparison_does_not_modify_old_prediction_artifacts(tmp_path: Path):
    compare_dir = tmp_path / "outputs" / "comparisons" / "20260521-174633"
    figures_dir = compare_dir / "figures"
    predictions_dir = compare_dir / "predictions"
    figures_dir.mkdir(parents=True)
    predictions_dir.mkdir()
    (compare_dir / "model_metrics.csv").write_text(
        "label,model,run_dir,test_rmse\n"
        "BNN-1h,improved_bnn,outputs/train/improved_bnn/20260521-170306,10.0\n",
        encoding="utf-8",
    )
    (predictions_dir / "BNN-1h.csv").write_text(
        "label,sample,horizon,target,mean,log_var,target_time\n"
        "BNN-1h,0,0,1.0,1.5,0.0,2020-01-01 08:00:00\n"
        "BNN-1h,0,1,5.0,1.0,0.0,2020-01-01 08:15:00\n",
        encoding="utf-8",
    )

    read_comparison("20260521-174633", tmp_path)

    unchanged_text = (predictions_dir / "BNN-1h.csv").read_text(encoding="utf-8")
    assert "lower_90" not in unchanged_text
    assert not (figures_dir / "prediction_0800_1200.png").exists()
    assert not (figures_dir / "prediction_window_metrics.csv").exists()


def test_list_train_runs_returns_training_outputs_only(tmp_path: Path):
    train_run = tmp_path / "outputs" / "train" / "improved_bnn" / "20260521-170306"
    train_run.mkdir(parents=True)
    (train_run / "config.yaml").write_text("model:\n  name: improved_bnn\n", encoding="utf-8")
    compare_dir = tmp_path / "outputs" / "comparisons" / "20260521-171302"
    compare_dir.mkdir(parents=True)
    (compare_dir / "model_metrics.csv").write_text("label,model,run_dir,test_rmse\nA,m,path,1.0\n", encoding="utf-8")

    runs = list_train_runs(tmp_path)

    assert [run["path"] for run in runs] == ["outputs/train/improved_bnn/20260521-170306"]


def test_create_comparison_uses_existing_compare_runner(tmp_path: Path):
    created_dir = tmp_path / "outputs" / "comparisons" / "20260521-180000"
    captured = {}

    def fake_runner(runs, name, output_dir, split, note):
        captured.update({"runs": runs, "name": name, "output_dir": output_dir, "split": split, "note": note})
        created_dir.mkdir(parents=True)
        (created_dir / "model_metrics.csv").write_text(
            "label,model,run_dir,test_rmse\nBNN-1h,improved_bnn,outputs/train/improved_bnn/1h,10.0\n",
            encoding="utf-8",
        )
        return created_dir

    payload = create_comparison(
        [{"label": "BNN-1h", "path": "outputs/train/improved_bnn/1h"}],
        tmp_path,
        name="visualizer",
        split="test",
        note="from ui",
        compare_runner=fake_runner,
    )

    assert captured == {
        "runs": [{"label": "BNN-1h", "path": "outputs/train/improved_bnn/1h"}],
        "name": "visualizer",
        "output_dir": tmp_path / "outputs",
        "split": "test",
        "note": "from ui",
    }
    assert payload["comparison"]["path"] == "outputs/comparisons/20260521-180000"
    assert payload["runs"][0]["label"] == "BNN-1h"


def test_create_comparison_with_real_runner_supports_zero_hour_bnn(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    _write_small_processed_splits(processed_dir)
    run_dir = run_training(_zero_hour_bnn_config(tmp_path, processed_dir))

    payload = create_comparison(
        [{"label": "BNN-0h", "path": str(run_dir.relative_to(tmp_path))}],
        tmp_path,
        name="visualizer",
        split="test",
        note="zero hour compare",
    )

    assert payload["comparison"]["runCount"] == 1
    assert payload["runs"][0]["label"] == "BNN-0h"
    assert payload["runs"][0]["predictionSummary"]["rows"] > 0


def test_ensure_project_root_on_syspath_inserts_project_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sys.path", [])

    ensure_project_root_on_syspath(tmp_path)

    assert str(tmp_path.resolve()) in __import__("sys").path


def test_visualizer_script_can_import_when_run_by_path():
    result = subprocess.run(
        [sys.executable, "visualizer/server.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Serve the BNN visualizer" in result.stdout
