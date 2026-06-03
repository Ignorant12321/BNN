"""Compare recursive 4h probabilistic interval methods from a trained run.

Usage:
    python -m src.experiments.compare_recursive_interval_methods_4h --run outputs/train/improved_bnn_recursive/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts.run_io import load_model_artifact, resolve_run_path, save_config, write_run_note
from src.data.pv import WindowArrays, load_split_window_arrays_from_config
from src.data.scaling import scaler_from_config, transform_window_arrays
from src.experiments.compare_bnn_strategies_4h import recursive_prediction_frame

INTERVAL_LEVELS = (90, 95)
NORMAL_Z = {
    90: 1.6448536269514722,
    95: 1.959963984540054,
}
PERSISTENCE_QUANTILES = {
    90: (0.05, 0.95),
    95: (0.025, 0.975),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare interval methods for a trained recursive 4h BNN run.")
    parser.add_argument("--run", required=True, help="Trained recursive 4h BNN run directory or model root.")
    parser.add_argument("--output-dir", default="outputs", help="Output root directory.")
    parser.add_argument("--note", default=None, help="Optional note for the comparison directory.")
    args = parser.parse_args()

    compare_dir = run_recursive_interval_comparison(args.run, output_dir=args.output_dir, note=args.note)
    print(f"Recursive interval comparison written to: {compare_dir}")
    print(f"Coverage summary: {compare_dir / 'coverage_summary.csv'}")


def run_recursive_interval_comparison(run: str | Path, output_dir: str | Path = "outputs", note: str | None = None) -> Path:
    """Load a trained recursive run and write interval-method comparison artifacts."""
    run_dir = resolve_run_path(run)
    model, step_config = load_model_artifact(run_dir)
    forecast_config = recursive_forecast_config(step_config)
    raw_arrays = load_split_window_arrays_from_config(forecast_config)
    scaler = scaler_from_config(step_config)
    model_arrays = {name: transform_window_arrays(arrays, scaler) for name, arrays in raw_arrays.items()} if scaler else raw_arrays

    val_predictions = recursive_prediction_frame("Our method", model, model_arrays["val"], config=step_config)
    test_predictions = recursive_prediction_frame("Our method", model, model_arrays["test"], config=step_config)
    val_normal_predictions = normal_residual_intervals(val_predictions, val_predictions)
    normal_predictions = normal_residual_intervals(val_predictions, test_predictions)
    val_persistence_predictions = persistence_interval_frame(raw_arrays["val"], raw_arrays["val"])
    persistence_predictions = persistence_interval_frame(raw_arrays["val"], raw_arrays["test"])
    rows = build_coverage_rows(test_predictions, normal_predictions, persistence_predictions)
    calibrated_rows = build_calibrated_rows(
        {
            "our_method": val_predictions,
            "normal": val_normal_predictions,
            "persistence": val_persistence_predictions,
        },
        {
            "our_method": test_predictions,
            "normal": normal_predictions,
            "persistence": persistence_predictions,
        },
    )

    compare_dir = create_recursive_interval_comparison_dir(output_dir)
    predictions_dir = compare_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    write_run_note(compare_dir, note)
    save_config(
        {
            "name": "recursive_interval_methods_4h",
            "source_run": str(run_dir),
            "output_dir": str(output_dir),
            "levels": list(INTERVAL_LEVELS),
        },
        compare_dir / "compare_config.yaml",
    )
    test_predictions.to_csv(predictions_dir / "our_method.csv", index=False)
    normal_predictions.to_csv(predictions_dir / "normal_distribution.csv", index=False)
    persistence_predictions.to_csv(predictions_dir / "persistence_interval.csv", index=False)
    write_coverage_summary(compare_dir / "coverage_summary.csv", rows)
    write_calibrated_summary(compare_dir / "calibrated_coverage_summary.csv", calibrated_rows)
    return compare_dir


def recursive_forecast_config(step_config: dict[str, Any]) -> dict[str, Any]:
    """Restore the full recursive forecast horizon from a saved one-step config."""
    result = dict(step_config)
    result["data"] = dict(step_config.get("data", {}))
    strategy = step_config.get("strategy", {})
    result["data"]["horizon"] = int(strategy.get("forecast_horizon", result["data"].get("horizon", 1)))
    return result


def normal_residual_intervals(
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    levels: tuple[int, ...] = INTERVAL_LEVELS,
) -> pd.DataFrame:
    """Build Gaussian residual intervals centered on recursive BNN point forecasts."""
    columns = [column for column in ("sample", "horizon", "target", "mean") if column in test_predictions.columns]
    result = test_predictions[columns].copy()
    val = val_predictions.copy()
    val["_residual"] = pd.to_numeric(val["target"], errors="coerce") - pd.to_numeric(val["mean"], errors="coerce")
    for level in levels:
        z_value = NORMAL_Z[int(level)]
        result[f"lower_{level}"] = np.nan
        result[f"upper_{level}"] = np.nan
        for horizon, group in val.groupby("horizon"):
            residuals = group["_residual"].dropna().to_numpy(dtype=float)
            residual_mean = float(np.mean(residuals)) if len(residuals) else 0.0
            residual_std = float(np.std(residuals, ddof=0)) if len(residuals) else 0.0
            mask = result["horizon"] == horizon
            center = result.loc[mask, "mean"].astype(float) + residual_mean
            result.loc[mask, f"lower_{level}"] = center - z_value * residual_std
            result.loc[mask, f"upper_{level}"] = center + z_value * residual_std
    return result


def persistence_interval_frame(
    val_arrays: WindowArrays,
    test_arrays: WindowArrays,
    levels: tuple[int, ...] = INTERVAL_LEVELS,
) -> pd.DataFrame:
    """Build empirical residual intervals around persistence point forecasts."""
    val_mean = persistence_mean(val_arrays)
    test_mean = persistence_mean(test_arrays)
    rows = []
    for sample_index in range(test_arrays.target.shape[0]):
        for horizon_index in range(test_arrays.target.shape[1]):
            rows.append(
                {
                    "sample": sample_index,
                    "horizon": horizon_index,
                    "target": float(test_arrays.target[sample_index, horizon_index]),
                    "mean": float(test_mean[sample_index, horizon_index]),
                }
            )
    result = pd.DataFrame(rows)
    residuals = val_arrays.target.astype(float) - val_mean.astype(float)
    for level in levels:
        lower_q, upper_q = PERSISTENCE_QUANTILES[int(level)]
        result[f"lower_{level}"] = np.nan
        result[f"upper_{level}"] = np.nan
        for horizon_index in range(residuals.shape[1]):
            horizon_residuals = residuals[:, horizon_index]
            lower_residual = float(np.quantile(horizon_residuals, lower_q))
            upper_residual = float(np.quantile(horizon_residuals, upper_q))
            mask = result["horizon"] == horizon_index
            center = result.loc[mask, "mean"].astype(float)
            result.loc[mask, f"lower_{level}"] = center + lower_residual
            result.loc[mask, f"upper_{level}"] = center + upper_residual
    if test_arrays.target_time is not None:
        result["target_time"] = [str(value) for row in test_arrays.target_time for value in row]
    return result


def persistence_mean(arrays: WindowArrays) -> np.ndarray:
    """Use previous-day same-time power, falling back to latest direct AC_POWER."""
    if arrays.direct.shape[1] < 1:
        raise ValueError("persistence interval method requires at least one direct feature")
    fallback = np.repeat(arrays.direct[:, :1].astype(float), arrays.target.shape[1], axis=1)
    if arrays.target_time is None:
        return fallback
    previous_day_lookup = previous_day_target_lookup(arrays)
    if not previous_day_lookup:
        return fallback
    result = fallback.copy()
    target_times = pd.to_datetime(np.asarray(arrays.target_time).reshape(-1), errors="coerce")
    for flat_index, target_time in enumerate(target_times):
        if pd.isna(target_time):
            continue
        previous_value = previous_day_lookup.get(target_time - pd.Timedelta(days=1))
        if previous_value is None:
            continue
        sample_index, horizon_index = divmod(flat_index, arrays.target.shape[1])
        result[sample_index, horizon_index] = previous_value
    return result


def previous_day_target_lookup(arrays: WindowArrays) -> dict[pd.Timestamp, float]:
    """Map target timestamps to observed target values for day-ahead persistence."""
    if arrays.target_time is None:
        return {}
    target_times = pd.to_datetime(np.asarray(arrays.target_time).reshape(-1), errors="coerce")
    target_values = np.asarray(arrays.target, dtype=float).reshape(-1)
    lookup: dict[pd.Timestamp, float] = {}
    for target_time, target_value in zip(target_times, target_values):
        if pd.isna(target_time) or not np.isfinite(target_value):
            continue
        lookup[target_time] = float(target_value)
    return lookup


def build_coverage_rows(
    our_predictions: pd.DataFrame,
    normal_predictions: pd.DataFrame,
    persistence_predictions: pd.DataFrame,
    levels: tuple[int, ...] = INTERVAL_LEVELS,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for level in levels:
        rows.append(
            {
                "confidence": int(level),
                "our_method_picp": interval_coverage(our_predictions, f"lower_{level}", f"upper_{level}"),
                "our_method_pinaw": interval_pinaw(our_predictions, f"lower_{level}", f"upper_{level}"),
                "normal_picp": interval_coverage(normal_predictions, f"lower_{level}", f"upper_{level}"),
                "normal_pinaw": interval_pinaw(normal_predictions, f"lower_{level}", f"upper_{level}"),
                "persistence_picp": interval_coverage(persistence_predictions, f"lower_{level}", f"upper_{level}"),
                "persistence_pinaw": interval_pinaw(persistence_predictions, f"lower_{level}", f"upper_{level}"),
            }
        )
    return rows


def interval_coverage(frame: pd.DataFrame, lower_column: str, upper_column: str) -> float:
    target = pd.to_numeric(frame["target"], errors="coerce")
    lower = pd.to_numeric(frame[lower_column], errors="coerce")
    upper = pd.to_numeric(frame[upper_column], errors="coerce")
    valid = target.notna() & lower.notna() & upper.notna()
    if not valid.any():
        return float("nan")
    covered = (target[valid] >= lower[valid]) & (target[valid] <= upper[valid])
    return float(covered.mean() * 100.0)


def interval_pinaw(frame: pd.DataFrame, lower_column: str, upper_column: str) -> float:
    """Return prediction interval normalized average width."""
    target = pd.to_numeric(frame["target"], errors="coerce")
    lower = pd.to_numeric(frame[lower_column], errors="coerce")
    upper = pd.to_numeric(frame[upper_column], errors="coerce")
    valid = target.notna() & lower.notna() & upper.notna()
    if not valid.any():
        return float("nan")
    width = upper[valid] - lower[valid]
    target_range = float(target[valid].max() - target[valid].min())
    scale = target_range if target_range > 0.0 else 1.0
    return float(width.mean() / scale)


def build_calibrated_rows(
    val_frames: dict[str, pd.DataFrame],
    test_frames: dict[str, pd.DataFrame],
    levels: tuple[int, ...] = INTERVAL_LEVELS,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for level in levels:
        target_coverage = float(level)
        for method, val_frame in val_frames.items():
            test_frame = test_frames[method]
            scale = calibration_scale_for_picp(val_frame, level, target_coverage)
            calibrated_val = scaled_interval_frame(val_frame, level, scale)
            calibrated_test = scaled_interval_frame(test_frame, level, scale)
            rows.append(
                {
                    "confidence": int(level),
                    "method": method,
                    "val_picp": interval_coverage(calibrated_val, f"lower_{level}", f"upper_{level}"),
                    "test_picp": interval_coverage(calibrated_test, f"lower_{level}", f"upper_{level}"),
                    "test_pinaw": interval_pinaw(calibrated_test, f"lower_{level}", f"upper_{level}"),
                    "test_pinaw_percent": interval_pinaw(calibrated_test, f"lower_{level}", f"upper_{level}") * 100.0,
                    "test_avg_width": interval_average_width(calibrated_test, f"lower_{level}", f"upper_{level}"),
                    "scale": scale,
                }
            )
    return rows


def calibration_scale_for_picp(frame: pd.DataFrame, level: int, target_coverage: float) -> float:
    target = pd.to_numeric(frame["target"], errors="coerce")
    lower = pd.to_numeric(frame[f"lower_{level}"], errors="coerce")
    upper = pd.to_numeric(frame[f"upper_{level}"], errors="coerce")
    valid = target.notna() & lower.notna() & upper.notna()
    if not valid.any():
        return float("nan")
    center = (lower[valid] + upper[valid]) / 2.0
    half_width = (upper[valid] - lower[valid]) / 2.0
    target_valid = target[valid]
    ratios = pd.Series(np.inf, index=target_valid.index, dtype=float)
    positive_width = half_width > 0.0
    ratios.loc[positive_width] = ((target_valid[positive_width] - center[positive_width]).abs() / half_width[positive_width]).to_numpy(dtype=float)
    ratios.loc[(~positive_width) & (target_valid == center)] = 0.0
    quantile = min(max(float(target_coverage) / 100.0, 0.0), 1.0)
    return float(max(0.0, np.quantile(ratios.to_numpy(dtype=float), quantile, method="higher")))


def scaled_interval_frame(frame: pd.DataFrame, level: int, scale: float) -> pd.DataFrame:
    result = frame.copy()
    lower_column = f"lower_{level}"
    upper_column = f"upper_{level}"
    lower = pd.to_numeric(result[lower_column], errors="coerce")
    upper = pd.to_numeric(result[upper_column], errors="coerce")
    center = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    result[lower_column] = center - scale * half_width
    result[upper_column] = center + scale * half_width
    return result


def interval_average_width(frame: pd.DataFrame, lower_column: str, upper_column: str) -> float:
    lower = pd.to_numeric(frame[lower_column], errors="coerce")
    upper = pd.to_numeric(frame[upper_column], errors="coerce")
    width = upper - lower
    width = width[np.isfinite(width)]
    if width.empty:
        return float("nan")
    return float(width.mean())


def write_coverage_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "confidence",
                "our_method_picp",
                "our_method_pinaw",
                "normal_picp",
                "normal_pinaw",
                "persistence_picp",
                "persistence_pinaw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_calibrated_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "confidence",
                "method",
                "val_picp",
                "test_picp",
                "test_pinaw",
                "test_pinaw_percent",
                "test_avg_width",
                "scale",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_recursive_interval_comparison_dir(output_dir: str | Path = "outputs") -> Path:
    compare_dir = Path(output_dir) / "comparisons" / f"recursive_interval_methods_4h_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    return compare_dir


if __name__ == "__main__":
    main()
