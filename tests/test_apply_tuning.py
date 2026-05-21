from __future__ import annotations

from pathlib import Path

import yaml

from src.experiments.apply_tuning import apply_best_params, collect_tuned_changes, format_changes


def test_collect_tuned_changes_uses_tuning_search_space(tmp_path: Path):
    base = tmp_path / "data.yaml"
    base.write_text("data:\n  lookback: 96\n  horizon: 16\n", encoding="utf-8")
    target = tmp_path / "bnn.yaml"
    target.write_text(
        "\n".join(
            [
                "include:",
                "  - data.yaml",
                "model:",
                "  name: improved_bnn",
                "  hidden_dim: 128",
                "training:",
                "  lr: 0.0005",
                "  kl_beta: 0.000001",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tuning_config = {
        "search_space": {
            "lr": {"type": "log_float"},
            "hidden_dim": {"type": "categorical"},
            "kl_beta": {"type": "log_float"},
        }
    }
    best_config = {
        "model": {"name": "improved_bnn", "hidden_dim": 64},
        "training": {"lr": 0.001, "kl_beta": 0.000001},
    }

    changes = collect_tuned_changes(target, tuning_config, best_config)

    assert [(item["path"], item["old"], item["new"]) for item in changes] == [
        ("training.lr", 0.0005, 0.001),
        ("model.hidden_dim", 128, 64),
    ]
    text = format_changes(changes)
    assert "training.lr" in text
    assert "0.0005" in text
    assert "0.001" in text


def test_apply_best_params_updates_target_only_after_confirmation(tmp_path: Path):
    base = tmp_path / "data.yaml"
    base.write_text("data:\n  lookback: 96\n  horizon: 16\n", encoding="utf-8")
    target = tmp_path / "bnn.yaml"
    target.write_text(
        "\n".join(
            [
                "include:",
                "  - data.yaml",
                "model:",
                "  name: improved_bnn",
                "  hidden_dim: 128",
                "training:",
                "  lr: 0.0005",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tuning_dir = tmp_path / "outputs" / "tuning" / "bnn_optuna"
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "tuning_config.yaml").write_text(
        yaml.safe_dump({"search_space": {"lr": {"type": "log_float"}, "hidden_dim": {"type": "categorical"}}}),
        encoding="utf-8",
    )
    (tuning_dir / "best_config.yaml").write_text(
        yaml.safe_dump({"model": {"name": "improved_bnn", "hidden_dim": 64}, "training": {"lr": 0.001}}),
        encoding="utf-8",
    )

    skipped = apply_best_params(tuning_dir, target, assume_yes=False, input_func=lambda _prompt: "n")
    assert skipped["applied"] is False
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["model"]["hidden_dim"] == 128

    applied = apply_best_params(tuning_dir, target, assume_yes=True)
    updated = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert applied["applied"] is True
    assert updated["include"] == ["data.yaml"]
    assert updated["model"]["hidden_dim"] == 64
    assert updated["training"]["lr"] == 0.001
