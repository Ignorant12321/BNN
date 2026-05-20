import csv
from pathlib import Path

import yaml

from src.experiments.compare_results import run_compare_results, run_compare_results_from_runs
from src.experiments.train import run_training


def _write_metrics(run_dir: Path, rmse: float, nll: float) -> None:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    with (metrics_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "rmse", "value": rmse})
        writer.writerow({"metric": "nll", "value": nll})


def test_compare_results_reads_existing_runs_without_training(tmp_path: Path):
    run_a = tmp_path / "outputs" / "improved_bnn" / "run-a"
    run_b = tmp_path / "outputs" / "cnn_baseline" / "run-b"
    _write_metrics(run_a, rmse=1.5, nll=2.5)
    _write_metrics(run_b, rmse=3.5, nll=4.5)
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "main",
                "output_dir": str(tmp_path / "outputs"),
                "runs": [
                    {"label": "BNN", "path": str(run_a)},
                    {"label": "CNN", "path": str(run_b)},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = run_compare_results(config_path)

    summary = (out_dir / "model_metrics.txt").read_text(encoding="utf-8")
    assert "BNN" in summary
    assert "CNN" in summary
    assert "1.5" in summary
    assert "4.5" in summary


def test_compare_results_accepts_single_cli_run_and_writes_txt(tmp_path: Path):
    run_a = tmp_path / "outputs" / "improved_bnn" / "run-a"
    _write_metrics(run_a, rmse=1.5, nll=2.5)

    out_dir = run_compare_results_from_runs(
        [{"label": "BNN-24h", "path": str(run_a)}],
        name="single",
        output_dir=tmp_path / "outputs",
    )

    summary_path = out_dir / "model_metrics.txt"
    assert summary_path.is_file()
    summary = summary_path.read_text(encoding="utf-8")
    assert "BNN-24h" in summary
    assert "improved_bnn" in summary
    assert "1.5" in summary
    assert "model_metrics.csv" not in summary


def test_compare_results_resolves_model_root_to_latest_timestamp_run(tmp_path: Path):
    model_root = tmp_path / "outputs" / "improved_bnn"
    old_run = model_root / "20260520-120000"
    latest_run = model_root / "20260520-130000"
    _write_metrics(old_run, rmse=9.5, nll=8.5)
    _write_metrics(latest_run, rmse=1.5, nll=2.5)

    out_dir = run_compare_results_from_runs(
        [{"label": "BNN-24h", "path": str(model_root)}],
        name="single",
        output_dir=tmp_path / "outputs",
    )

    summary = (out_dir / "model_metrics.txt").read_text(encoding="utf-8")
    assert str(latest_run) in summary
    assert "1.5" in summary
    assert "9.5" not in summary


def test_compare_loads_trained_run_and_evaluates_test_split(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for split_name, values in {
        "train": [1.0, 2.0, 3.0, 4.0],
        "val": [10.0, 11.0, 12.0, 13.0],
        "test": [20.0, 21.0, 22.0, 23.0],
    }.items():
        frame = {
            "DATE_TIME": [f"2020-01-01 00:{minute:02d}:00" for minute in range(0, 60, 15)],
            "AC_POWER": values,
            "AMBIENT_TEMPERATURE": [25.0] * 4,
            "MODULE_TEMPERATURE": [30.0] * 4,
            "IRRADIATION": [0.5] * 4,
        }
        import pandas as pd

        pd.DataFrame(frame).to_csv(processed_dir / f"{split_name}.csv", index=False)
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 2,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "mlp_baseline"},
        "training": {"backend": "numpy"},
    }
    run_dir = run_training(config)

    out_dir = run_compare_results_from_runs(
        [{"label": "MLP", "path": str(run_dir)}],
        name="main",
        output_dir=tmp_path / "outputs",
    )

    assert out_dir.parent.parent.name == "comparisons"
    assert (out_dir / "compare_config.yaml").is_file()
    assert (out_dir / "model_metrics.csv").is_file()
    assert (out_dir / "model_metrics.txt").is_file()
    assert (out_dir / "predictions" / "MLP.csv").is_file()
    assert (out_dir / "figures" / "metrics_bar.svg").is_file()
    assert (out_dir / "report.md").is_file()
    summary = (out_dir / "model_metrics.txt").read_text(encoding="utf-8")
    assert "test_rmse" in summary


def test_compare_evaluates_trained_torch_run(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    import pandas as pd

    for split_name, values in {
        "train": [1.0, 2.0, 3.0, 4.0],
        "val": [10.0, 11.0, 12.0, 13.0],
        "test": [20.0, 21.0, 22.0, 23.0],
    }.items():
        pd.DataFrame(
            {
                "DATE_TIME": [f"2020-01-01 00:{minute:02d}:00" for minute in range(0, 60, 15)],
                "AC_POWER": values,
                "AMBIENT_TEMPERATURE": [25.0] * 4,
                "MODULE_TEMPERATURE": [30.0] * 4,
                "IRRADIATION": [0.5] * 4,
            }
        ).to_csv(processed_dir / f"{split_name}.csv", index=False)
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 2,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "mlp_baseline", "hidden_dim": 8, "branch_dim": 4},
        "training": {"backend": "torch", "device": "auto", "epochs": 1, "batch_size": 2, "lr": 0.01},
    }
    run_dir = run_training(config)

    out_dir = run_compare_results_from_runs(
        [{"label": "Torch-MLP", "path": str(run_dir)}],
        name="torch",
        output_dir=tmp_path / "outputs",
    )

    assert (out_dir / "predictions" / "Torch-MLP.csv").is_file()
    summary = (out_dir / "model_metrics.txt").read_text(encoding="utf-8")
    assert "test_rmse" in summary
