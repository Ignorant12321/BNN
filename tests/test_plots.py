from pathlib import Path

import pandas as pd

import src.evaluation.plots as plots


def test_default_prediction_intervals_cover_6_to_18_daytime_windows():
    assert plots.DEFAULT_PREDICTION_INTERVALS == (("06:00", "10:00"), ("10:00", "14:00"), ("14:00", "18:00"))


def test_training_loss_plot_does_not_overlay_metric_notes(monkeypatch):
    captured = {}

    def fake_write_line_png(series, path, title, notes, markers=None, x_label=None, y_label=None):
        captured["series"] = series
        captured["path"] = path
        captured["title"] = title
        captured["notes"] = notes
        captured["x_label"] = x_label
        captured["y_label"] = y_label

    monkeypatch.setattr(plots, "write_line_png", fake_write_line_png)

    plots.write_training_loss_png(
        [{"epoch": 1, "loss": 2.0}, {"epoch": 2, "loss": 1.0}],
        {"test_rmse": 3.0, "test_mae": 2.0},
        Path("loss_curve.png"),
    )

    assert captured["title"] == "模型训练损失变化曲线"
    assert captured["notes"] == []
    assert captured["x_label"] == "迭代次数"
    assert captured["y_label"] == "损失值"


def test_training_loss_plot_includes_validation_loss_and_epoch_markers(monkeypatch):
    captured = {}

    def fake_write_line_png(series, path, title, notes, markers=None, x_label=None, y_label=None):
        captured["series"] = series
        captured["markers"] = markers

    monkeypatch.setattr(plots, "write_line_png", fake_write_line_png)

    plots.write_training_loss_png(
        [
            {"epoch": 1, "loss": 2.0, "val_loss": 2.5},
            {"epoch": 2, "loss": 1.0, "val_loss": 1.2},
            {"epoch": 3, "loss": 0.8, "val_loss": 1.4, "early_stop": 1.0},
        ],
        {},
        Path("loss_curve.png"),
        best_epoch=2,
    )

    assert [item["label"] for item in captured["series"]] == ["train loss", "val loss"]
    assert captured["series"][1]["points"] == [(1.0, 2.5), (2.0, 1.2), (3.0, 1.4)]
    assert captured["markers"] == [
        {"x": 2.0, "label": "best epoch", "color": "#16a34a", "linestyle": "--"},
        {"x": 3.0, "label": "early stop", "color": "#dc2626", "linestyle": ":"},
    ]


def test_prediction_window_plot_labels_axes_and_interval_legend(monkeypatch, tmp_path):
    calls = {"fill_labels": [], "xlabel": None, "ylabel": None}

    class FakeAxis:
        transAxes = object()
        xaxis = type("XAxis", (), {"set_major_formatter": lambda self, formatter: None})()

        def fill_between(self, *args, **kwargs):
            calls["fill_labels"].append(kwargs.get("label"))

        def plot(self, *args, **kwargs):
            pass

        def set_title(self, title):
            pass

        def set_xlabel(self, label):
            calls["xlabel"] = label

        def set_ylabel(self, label):
            calls["ylabel"] = label

        def grid(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

        def text(self, *args, **kwargs):
            pass

    class FakeFigure:
        def autofmt_xdate(self):
            pass

        def tight_layout(self):
            pass

        def savefig(self, path):
            Path(path).write_bytes(b"\x89PNG\r\n")

    monkeypatch.setattr(plots.plt, "subplots", lambda *args, **kwargs: (FakeFigure(), FakeAxis()))
    monkeypatch.setattr(plots.plt, "close", lambda fig: None)

    frame = pd.DataFrame(
        {
            "label": ["BNN"],
            "sample": [0],
            "horizon": [0],
            "target_time": ["2020-01-01 08:15"],
            "target": [1.0],
            "mean": [1.1],
            "log_var": [0.0],
        }
    )

    plots.write_prediction_window_png(frame, tmp_path / "prediction.png", "08:00", "12:00")

    assert calls["xlabel"] == "Time"
    assert calls["ylabel"] == "AC Power / kW"
    assert "BNN 95% interval" in calls["fill_labels"]
    assert "BNN 90% interval" in calls["fill_labels"]


def test_prediction_window_plot_can_hide_interval_bands(monkeypatch, tmp_path):
    calls = {"fill_count": 0}

    class FakeAxis:
        transAxes = object()
        xaxis = type("XAxis", (), {"set_major_formatter": lambda self, formatter: None})()

        def fill_between(self, *args, **kwargs):
            calls["fill_count"] += 1

        def plot(self, *args, **kwargs):
            pass

        def set_title(self, title):
            pass

        def set_xlabel(self, label):
            pass

        def set_ylabel(self, label):
            pass

        def grid(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

        def text(self, *args, **kwargs):
            pass

    class FakeFigure:
        def autofmt_xdate(self):
            pass

        def tight_layout(self):
            pass

        def savefig(self, path):
            Path(path).write_bytes(b"\x89PNG\r\n")

    monkeypatch.setattr(plots.plt, "subplots", lambda *args, **kwargs: (FakeFigure(), FakeAxis()))
    monkeypatch.setattr(plots.plt, "close", lambda fig: None)

    frame = pd.DataFrame(
        {
            "label": ["BNN"],
            "sample": [0],
            "horizon": [0],
            "target_time": ["2020-01-01 08:15"],
            "target": [1.0],
            "mean": [1.1],
            "log_var": [0.0],
        }
    )

    plots.write_prediction_window_png(
        frame,
        tmp_path / "prediction.png",
        "08:00",
        "12:00",
        show_intervals=False,
    )

    assert calls["fill_count"] == 0


def test_prediction_window_subset_returns_single_forecast_trajectory_from_start_clock():
    frame = pd.DataFrame(
        {
            "label": ["BNN", "BNN", "BNN", "BNN", "BNN"],
            "sample": [0, 0, 0, 1, 1],
            "horizon": [0, 1, 15, 0, 15],
            "target_time": [
                "2020-01-01 08:15",
                "2020-01-01 08:30",
                "2020-01-01 12:00",
                "2020-01-01 08:00",
                "2020-01-01 11:45",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mean": [1.1, 2.1, 3.1, 4.1, 5.1],
            "log_var": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    subset = plots.prediction_window_subset(frame, "08:00", "12:00")

    assert subset["sample"].tolist() == [0, 0, 0]
    assert subset["horizon"].tolist() == [0, 1, 15]
    assert subset["target_time"].tolist() == [
        "2020-01-01 08:15",
        "2020-01-01 08:30",
        "2020-01-01 12:00",
    ]


def test_prediction_window_subset_falls_back_to_next_date_with_matching_issue_time():
    frame = pd.DataFrame(
        {
            "label": ["BNN", "BNN", "BNN"],
            "sample": [0, 1, 1],
            "horizon": [0, 0, 15],
            "target_time": [
                "2020-01-01 08:30",
                "2020-01-02 08:15",
                "2020-01-02 12:00",
            ],
            "target": [1.0, 2.0, 3.0],
            "mean": [1.1, 2.1, 3.1],
            "log_var": [0.0, 0.0, 0.0],
        }
    )

    subset = plots.prediction_window_subset(frame, "08:00", "12:00")

    assert subset["sample"].tolist() == [1, 1]
    assert subset["target_time"].tolist() == [
        "2020-01-02 08:15",
        "2020-01-02 12:00",
    ]
