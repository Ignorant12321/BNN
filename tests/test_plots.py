from pathlib import Path

import src.evaluation.plots as plots


def test_training_loss_plot_does_not_overlay_metric_notes(monkeypatch):
    captured = {}

    def fake_write_line_png(series, path, title, notes, markers=None):
        captured["series"] = series
        captured["path"] = path
        captured["title"] = title
        captured["notes"] = notes

    monkeypatch.setattr(plots, "write_line_png", fake_write_line_png)

    plots.write_training_loss_png(
        [{"epoch": 1, "loss": 2.0}, {"epoch": 2, "loss": 1.0}],
        {"test_rmse": 3.0, "test_mae": 2.0},
        Path("loss_curve.png"),
    )

    assert captured["title"] == "Training Loss"
    assert captured["notes"] == []


def test_training_loss_plot_includes_validation_loss_and_epoch_markers(monkeypatch):
    captured = {}

    def fake_write_line_png(series, path, title, notes, markers=None):
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
