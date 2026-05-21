"""PNG 图表输出。"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PREDICTION_INTERVALS = (("08:00", "12:00"), ("10:00", "14:00"), ("12:00", "16:00"))
LINE_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def write_metrics_bar_png(rows: list[dict[str, str]], path: Path, metric: str = "test_rmse") -> None:
    """写出单个指标的模型对比柱状图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for row in rows:
        try:
            value = float(row.get(metric, row.get(metric.removeprefix("test_"), "nan")))
        except ValueError:
            continue
        if isfinite(value):
            values.append((row["label"], value))

    fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.45 * max(len(values), 1) + 1.2)), dpi=140)
    if values:
        labels = [label for label, _ in values]
        metric_values = [value for _, value in values]
        ax.barh(labels, metric_values, color="#4f8cc9")
        ax.invert_yaxis()
        for index, value in enumerate(metric_values):
            ax.text(value, index, f" {value:.4f}", va="center", fontsize=8)
        ax.set_xlabel(metric)
    else:
        ax.text(0.5, 0.5, "no metric data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_training_loss_png(epoch_history: list[dict[str, float]], metrics: dict[str, float], path: Path) -> None:
    """写出单个训练产物的 loss 曲线，并在图中附带训练/验证指标。"""
    series = []
    if epoch_history:
        series.append(
            {
                "label": "loss",
                "color": LINE_COLORS[0],
                "points": [(float(item["epoch"]), float(item["loss"])) for item in epoch_history],
            }
        )
    notes = [f"{name}: {value:.4f}" for name, value in metrics.items() if isfinite(float(value))]
    write_line_png(series, path, title="Training Loss", notes=notes[:10])


def write_comparison_loss_png(histories: list[dict], path: Path) -> None:
    """把多个训练产物的 epoch loss 画到同一张图。"""
    series = []
    for index, item in enumerate(histories):
        history = item.get("history", [])
        if not history:
            continue
        series.append(
            {
                "label": str(item["label"]),
                "color": LINE_COLORS[index % len(LINE_COLORS)],
                "points": [(float(row["epoch"]), float(row["loss"])) for row in history],
            }
        )
    write_line_png(series, path, title="Training Loss Comparison", notes=[])


def write_prediction_window_pngs(
    prediction_frames: list[pd.DataFrame],
    figures_dir: Path,
    intervals: tuple[tuple[str, str], ...] = DEFAULT_PREDICTION_INTERVALS,
) -> None:
    """按固定时段写出测试集某一天的真实值/预测值曲线。"""
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    for start, end in intervals:
        path = figures_dir / f"prediction_{_interval_slug(start)}_{_interval_slug(end)}.png"
        write_prediction_window_png(combined, path, start, end)


def write_prediction_window_png(frame: pd.DataFrame, path: Path, start: str, end: str) -> None:
    """写出一个时间段的预测曲线。"""
    title = f"Prediction {start}-{end}"
    if frame.empty or "target_time" not in frame.columns:
        write_line_png([], path, title=title, notes=["no target_time data"])
        return

    work = frame.copy()
    work["_target_time"] = pd.to_datetime(work["target_time"], errors="coerce")
    work = work.dropna(subset=["_target_time"])
    if work.empty:
        write_line_png([], path, title=title, notes=["no target_time data"])
        return

    start_minute = _time_to_minutes(start)
    end_minute = _time_to_minutes(end)
    work["_minute"] = work["_target_time"].dt.hour * 60 + work["_target_time"].dt.minute
    subset = pd.DataFrame()
    for date_value in sorted(work["_target_time"].dt.date.unique()):
        candidate = work[
            (work["_target_time"].dt.date == date_value)
            & (work["_minute"] >= start_minute)
            & (work["_minute"] < end_minute)
        ]
        if not candidate.empty:
            subset = candidate
            break
    if subset.empty:
        write_line_png([], path, title=title, notes=["no rows in interval"])
        return

    series = []
    target = subset.groupby("_target_time", as_index=False)["target"].mean().sort_values("_target_time")
    series.append(
        {
            "label": "actual",
            "color": "#111827",
            "points": [(row["_target_time"], float(row["target"])) for _, row in target.iterrows()],
        }
    )
    labels = sorted(str(label) for label in subset["label"].dropna().unique())
    for index, label in enumerate(labels):
        predicted = (
            subset[subset["label"] == label]
            .groupby("_target_time", as_index=False)["mean"]
            .mean()
            .sort_values("_target_time")
        )
        series.append(
            {
                "label": label,
                "color": LINE_COLORS[index % len(LINE_COLORS)],
                "points": [(row["_target_time"], float(row["mean"])) for _, row in predicted.iterrows()],
            }
        )
    date_note = str(subset["_target_time"].dt.date.iloc[0])
    write_line_png(series, path, title=title, notes=[date_note])


def write_line_png(series: list[dict], path: Path, title: str, notes: list[str]) -> None:
    """通用折线 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=140)
    has_points = False
    for item in series:
        points = item.get("points", [])
        if not points:
            continue
        has_points = True
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.plot(x_values, y_values, label=str(item.get("label", "series")), color=item.get("color"), linewidth=1.8)

    ax.set_title(title)
    if has_points:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        first_x = next(point[0] for item in series for point in item.get("points", []))
        if isinstance(first_x, pd.Timestamp):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, "no curve data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    for index, note in enumerate(notes):
        ax.text(0.99, 0.96 - index * 0.06, note, ha="right", va="top", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _time_to_minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def _interval_slug(value: str) -> str:
    return value.replace(":", "")
