from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.compare import expand_run_entries, format_summary_table, run_compare_from_runs
from src.experiments.train import run_training


METRIC_NAMES = [
    "test_mae",
    "test_rmse",
    "test_nmae",
    "test_nrmse",
    "test_picp_90",
    "test_pinaw_90",
    "test_picp_95",
    "test_pinaw_95",
]


def _write_processed_splits(processed_dir: Path) -> None:
    processed_dir.mkdir()
    times = pd.date_range("2020-01-01 07:00:00", periods=40, freq="15min")
    base = {
        "DATE_TIME": times,
        "AC_POWER": np.linspace(1.0, 40.0, len(times), dtype=np.float32),
        "AMBIENT_TEMPERATURE": np.linspace(20.0, 30.0, len(times), dtype=np.float32),
        "MODULE_TEMPERATURE": np.linspace(25.0, 35.0, len(times), dtype=np.float32),
        "IRRADIATION": np.linspace(0.1, 0.9, len(times), dtype=np.float32),
    }
    for split_name in ("train", "val", "test"):
        pd.DataFrame(base).to_csv(processed_dir / f"{split_name}.csv", index=False)


def _training_config(tmp_path: Path, processed_dir: Path) -> dict:
    return {
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "generation_path": str(tmp_path / "missing_generation.csv"),
            "weather_path": str(tmp_path / "missing_weather.csv"),
            "processed_dir": str(processed_dir),
            "lookback": 1,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "mlp_baseline"},
        "training": {"backend": "numpy", "device": "auto"},
    }


def test_compare_outputs_research_artifacts_without_duplicate_txt(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    _write_processed_splits(processed_dir)
    run_dir = run_training(_training_config(tmp_path, processed_dir))

    out_dir = run_compare_from_runs(
        [{"label": "MLP", "path": str(run_dir)}],
        name="comparison",
        output_dir=tmp_path / "outputs",
    )

    assert out_dir.parent == tmp_path / "outputs" / "comparisons"
    assert (out_dir / "compare_config.yaml").is_file()
    assert (out_dir / "note.txt").read_text(encoding="utf-8") == f"{out_dir.name}\n"
    assert (out_dir / "model_metrics.csv").is_file()
    assert not (out_dir / "model_metrics.txt").exists()
    assert (out_dir / "predictions" / "MLP.csv").is_file()
    assert not (out_dir / "report.md").exists()

    metrics_text = (out_dir / "model_metrics.csv").read_text(encoding="utf-8")
    for metric_name in METRIC_NAMES:
        assert metric_name in metrics_text
        assert not (out_dir / "figures" / f"metrics_{metric_name}.png").exists()
    assert "nll" not in metrics_text

    predictions_text = (out_dir / "predictions" / "MLP.csv").read_text(encoding="utf-8")
    assert "target_time" in predictions_text
    assert "lower_90" in predictions_text
    assert "upper_90" in predictions_text
    assert "lower_95" in predictions_text
    assert "upper_95" in predictions_text
    assert (out_dir / "figures" / "loss_curves.png").read_bytes().startswith(b"\x89PNG")
    assert (out_dir / "figures" / "prediction_0800_1200.png").read_bytes().startswith(b"\x89PNG")
    assert (out_dir / "figures" / "prediction_1000_1400.png").read_bytes().startswith(b"\x89PNG")
    assert (out_dir / "figures" / "prediction_1200_1600.png").read_bytes().startswith(b"\x89PNG")
    assert (out_dir / "figures" / "prediction_window_metrics.csv").is_file()


def test_compare_writes_note_file_from_argument(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    _write_processed_splits(processed_dir)
    run_dir = run_training(_training_config(tmp_path, processed_dir))

    out_dir = run_compare_from_runs(
        [{"label": "MLP", "path": str(run_dir)}],
        name="comparison",
        output_dir=tmp_path / "outputs",
        note="best model comparison",
    )

    assert (out_dir / "note.txt").read_text(encoding="utf-8") == "best model comparison\n"


def test_expand_run_entries_expands_model_root_to_child_run_dirs(tmp_path: Path):
    model_root = tmp_path / "outputs" / "train" / "improved_bnn"
    for run_name in ("20260521-110241", "renamed-4h"):
        run_dir = model_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.csv").write_text("split,metric,value\nval,rmse,1.0\n", encoding="utf-8")
    (model_root / "notes").mkdir()

    runs = expand_run_entries([{"path": str(model_root)}])

    assert runs == [
        {"label": "20260521-110241", "path": str(model_root / "20260521-110241")},
        {"label": "renamed-4h", "path": str(model_root / "renamed-4h")},
    ]


def test_expand_run_entries_keeps_explicit_single_run_label(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "train" / "improved_bnn" / "renamed-24h"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  name: improved_bnn\n", encoding="utf-8")

    runs = expand_run_entries([{"label": "BNN-24h", "path": str(run_dir)}])

    assert runs == [{"label": "BNN-24h", "path": str(run_dir)}]


def test_format_summary_table_prints_one_run_per_block():
    rows = [
        {
            "label": "BNN-1h",
            "model": "improved_bnn",
            "run_dir": "outputs/train/improved_bnn/20260521-170306",
            "test_mae": "1021.7421644445051",
            "test_rmse": "1705.6773255172839",
        },
        {
            "label": "BNN-4h",
            "model": "improved_bnn",
            "run_dir": "outputs/train/improved_bnn/20260521-170502",
            "test_mae": "1014.672523704735",
            "test_rmse": "1720.8247427233955",
        },
    ]

    text = format_summary_table(rows)

    assert "Run 1: BNN-1h" in text
    assert "Run 2: BNN-4h" in text
    assert "test_mae  : 1021.7421644445051" in text
    assert "test_rmse : 1720.8247427233955" in text
    assert "label | model | run_dir" not in text
