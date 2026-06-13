"""PNG 图表输出。"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from src.evaluation.metrics import BASE_METRIC_NAMES, regression_metrics


DEFAULT_PREDICTION_INTERVALS = (("06:00", "10:00"), ("10:00", "14:00"), ("14:00", "18:00"))
LINE_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
CHINESE_FONT_CANDIDATES = ("Microsoft YaHei", "Noto Sans SC", "SimHei", "SimSun")


def write_training_loss_png(
    epoch_history: list[dict[str, float]],
    metrics: dict[str, float],
    path: Path,
    best_epoch: int | None = None,
) -> None:
    """写出单个训练产物的 loss 曲线。"""
    series = []
    if epoch_history:
        series.append(
            {
                "label": "train loss",
                "color": LINE_COLORS[0],
                "points": [(float(item["epoch"]), float(item["loss"])) for item in epoch_history],
            }
        )
        val_points = [(float(item["epoch"]), float(item["val_loss"])) for item in epoch_history if "val_loss" in item]
        if val_points:
            series.append({"label": "val loss", "color": LINE_COLORS[1], "points": val_points})
    markers = []
    if best_epoch is not None:
        markers.append({"x": float(best_epoch), "label": "best epoch", "color": "#16a34a", "linestyle": "--"})
    early_stop_epoch = first_marked_epoch(epoch_history, "early_stop")
    if early_stop_epoch is not None:
        markers.append({"x": early_stop_epoch, "label": "early stop", "color": "#dc2626", "linestyle": ":"})
    write_line_png(
        series,
        path,
        title="模型训练损失变化曲线",
        notes=[],
        markers=markers,
        x_label="迭代次数",
        y_label="损失值",
    )


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
    show_intervals: bool = True,
) -> None:
    """按固定时段写出测试集某一天的真实值/预测值曲线。"""
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    for start, end in intervals:
        path = figures_dir / f"prediction_{_interval_slug(start)}_{_interval_slug(end)}.png"
        write_prediction_window_png(combined, path, start, end, show_intervals=show_intervals)


def write_prediction_window_metrics_csv(
    prediction_frame: pd.DataFrame,
    path: Path,
    intervals: tuple[tuple[str, str], ...] = DEFAULT_PREDICTION_INTERVALS,
) -> None:
    """Write per-interval metrics for the same windows used by prediction plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["interval_start", "interval_end", *BASE_METRIC_NAMES]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for start, end in intervals:
            row = {"interval_start": start, "interval_end": end}
            row.update(prediction_window_metrics(prediction_frame, start, end))
            writer.writerow(row)


def prediction_window_metrics(frame: pd.DataFrame, start: str, end: str) -> dict[str, float | str]:
    """Calculate metrics for the first available forecast trajectory issued at start."""
    subset = prediction_window_subset(frame, start, end)
    if subset.empty:
        return {name: "" for name in BASE_METRIC_NAMES}
    metrics = regression_metrics(
        subset["mean"].to_numpy(dtype=float).reshape(-1, 1),
        subset["log_var"].to_numpy(dtype=float).reshape(-1, 1),
        subset["target"].to_numpy(dtype=float).reshape(-1, 1),
    )
    return {name: metrics[name] for name in BASE_METRIC_NAMES}


def write_prediction_window_png(
    frame: pd.DataFrame,
    path: Path,
    start: str,
    end: str,
    show_intervals: bool = True,
) -> None:
    """写出一个起报时刻对应的未来预测曲线。"""
    title = f"Prediction {start}-{end}"
    if frame.empty or "target_time" not in frame.columns:
        write_line_png([], path, title=title, notes=["no target_time data"])
        return

    subset = prediction_window_subset(frame, start, end)
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
        predicted = prediction_interval_bounds(subset, label)
        ax_color = LINE_COLORS[index % len(LINE_COLORS)]
        series.append(
            {
                "label": label,
                "color": ax_color,
                "points": [(row["_target_time"], float(row["mean"])) for _, row in predicted.iterrows()],
                "intervals": predicted if show_intervals else None,
            }
        )
    date_note = str(subset["_target_time"].dt.date.iloc[0])
    write_line_png(series, path, title=title, notes=[date_note])


def prediction_window_subset(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows for the first forecast issued at start and ending at end."""
    if frame.empty or "target_time" not in frame.columns or "horizon" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["_target_time"] = pd.to_datetime(work["target_time"], errors="coerce")
    work["_horizon"] = pd.to_numeric(work["horizon"], errors="coerce")
    work = work.dropna(subset=["_target_time", "_horizon"])
    if work.empty:
        return pd.DataFrame()
    start_minute = _time_to_minutes(start)
    end_minute = _time_to_minutes(end)
    work["_issue_time"] = work["_target_time"] - pd.to_timedelta((work["_horizon"] + 1) * 15, unit="m")
    for date_value in sorted(work["_issue_time"].dt.date.unique()):
        issue_time = pd.Timestamp(date_value) + pd.Timedelta(minutes=start_minute)
        end_time = pd.Timestamp(date_value) + pd.Timedelta(minutes=end_minute)
        candidate = work[
            (work["_issue_time"] == issue_time)
            & (work["_target_time"] > issue_time)
            & (work["_target_time"] <= end_time)
        ]
        if not candidate.empty:
            sort_columns = ["_target_time"]
            if "label" in candidate.columns:
                sort_columns = ["label", *sort_columns]
            return candidate.sort_values(sort_columns)
    return pd.DataFrame()


def prediction_interval_bounds(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return mean and 90/95 interval bounds for one label grouped by target time."""
    work = frame[frame["label"].astype(str) == str(label)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["_target_time", "mean", "std", "lower_90", "upper_90", "lower_95", "upper_95"]
        )
    if "_target_time" not in work.columns:
        work["_target_time"] = pd.to_datetime(work["target_time"], errors="coerce")
    work = work.dropna(subset=["_target_time"])
    work["_mean"] = pd.to_numeric(work["mean"], errors="coerce")
    work["_variance"] = np.exp(pd.to_numeric(work["log_var"], errors="coerce"))
    work = work.dropna(subset=["_mean"])
    if work.empty:
        return pd.DataFrame(
            columns=["_target_time", "mean", "std", "lower_90", "upper_90", "lower_95", "upper_95"]
        )

    rows = []
    for target_time, group in work.groupby("_target_time"):
        mean_values = group["_mean"].to_numpy(dtype=float)
        variance_values = group["_variance"].to_numpy(dtype=float)
        mean = float(np.mean(mean_values))
        finite_variance = np.isfinite(variance_values)
        if np.any(finite_variance):
            variance = float(np.mean(variance_values[finite_variance] + mean_values[finite_variance] ** 2) - mean**2)
            std = float(np.sqrt(max(variance, 0.0)))
        else:
            std = float("nan")
        rows.append(
            {
                "_target_time": target_time,
                "mean": mean,
                "std": std,
                "lower_90": mean - 1.6448536269514722 * std,
                "upper_90": mean + 1.6448536269514722 * std,
                "lower_95": mean - 1.959963984540054 * std,
                "upper_95": mean + 1.959963984540054 * std,
            }
        )
    return pd.DataFrame(rows).sort_values("_target_time").reset_index(drop=True)


def write_line_png(
    series: list[dict],
    path: Path,
    title: str,
    notes: list[str],
    markers: list[dict] | None = None,
    x_label: str = "Time",
    y_label: str = "AC Power / kW",
) -> None:
    """通用折线 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=140)
    text_kwargs = text_font_kwargs(title, x_label, y_label, *notes)
    has_points = False
    for item in series:
        points = item.get("points", [])
        if not points:
            continue
        has_points = True
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        intervals = item.get("intervals")
        if isinstance(intervals, pd.DataFrame) and not intervals.empty:
            interval_x = intervals["_target_time"]
            color = item.get("color")
            label = str(item.get("label", "series"))
            ax.fill_between(
                interval_x,
                intervals["lower_95"],
                intervals["upper_95"],
                color=color,
                alpha=0.10,
                linewidth=0,
                label=f"{label} 95% interval",
            )
            ax.fill_between(
                interval_x,
                intervals["lower_90"],
                intervals["upper_90"],
                color=color,
                alpha=0.18,
                linewidth=0,
                label=f"{label} 90% interval",
            )
        ax.plot(x_values, y_values, label=str(item.get("label", "series")), color=item.get("color"), linewidth=1.8)

    ax.set_title(title, **text_kwargs)
    if has_points:
        ax.set_xlabel(x_label, **text_kwargs)
        ax.set_ylabel(y_label, **text_kwargs)
        ax.grid(True, alpha=0.25)
        for marker in markers or []:
            ax.axvline(
                float(marker["x"]),
                color=marker.get("color"),
                linestyle=marker.get("linestyle", "--"),
                linewidth=1.3,
                alpha=0.85,
                label=str(marker.get("label", "")),
            )
        ax.legend(loc="best", fontsize=8)
        first_x = next(point[0] for item in series for point in item.get("points", []))
        if isinstance(first_x, pd.Timestamp):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, "no curve data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    for index, note in enumerate(notes):
        ax.text(
            0.99,
            0.96 - index * 0.06,
            note,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            **text_kwargs,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def first_marked_epoch(epoch_history: list[dict[str, float]], marker_name: str) -> float | None:
    for item in epoch_history:
        if marker_name in item:
            return float(item["epoch"])
    return None


def text_font_kwargs(*texts: object) -> dict[str, str]:
    """Return a CJK-capable font only when the requested text needs one."""
    if not any(contains_cjk(text) for text in texts):
        return {}
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in available:
            return {"fontfamily": font_name}
    return {}


def contains_cjk(text: object) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


def _time_to_minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def _interval_slug(value: str) -> str:
    return value.replace(":", "")
