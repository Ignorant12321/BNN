from pathlib import Path

import pytest
import yaml

from src.config import load_config


def test_load_config_reads_yaml_file(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"name": "mlp"}, "training": {"epochs": 2}}), encoding="utf-8")

    config = load_config(config_path)

    assert config["model"]["name"] == "mlp"
    assert config["training"]["epochs"] == 2


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level YAML object must be a mapping"):
        load_config(config_path)


def test_load_config_supports_include_defaults_and_overrides(tmp_path: Path):
    base_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "lookback": 96,
                    "horizon": 16,
                    "features": {
                        "history": ["AC_POWER"],
                        "weather": ["IRRADIATION"],
                        "direct": ["AC_POWER"],
                        "target": "AC_POWER",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    model_path.write_text(
        yaml.safe_dump(
            {
                "include": ["data.yaml"],
                "data": {"lookback": 4},
                "model": {"name": "improved_bnn"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(model_path)

    assert config["data"]["lookback"] == 4
    assert config["data"]["horizon"] == 16
    assert config["training"]["backend"] == "torch"
    assert config["training"]["device"] == "auto"
    assert config["training"]["weight_decay"] == 0.0
    assert config["output_dir"] == "outputs"


def test_project_model_configs_use_nested_defaults():
    expected = {
        "configs/models/bnn/1h.yaml": ("improved_bnn", 4),
        "configs/models/bnn/4h.yaml": ("improved_bnn", 16),
        "configs/models/bnn/8h.yaml": ("improved_bnn", 32),
        "configs/models/bnn/12h.yaml": ("improved_bnn", 48),
        "configs/models/bnn/24h.yaml": ("improved_bnn", 96),
        "configs/models/mlp/24h.yaml": ("mlp_baseline", 96),
        "configs/models/cnn/24h.yaml": ("cnn_baseline", 96),
        "configs/models/mc_dropout/24h.yaml": ("mc_dropout", 96),
    }

    for config_path, (model_name, lookback) in expected.items():
        config = load_config(config_path)

        assert config["model"]["name"] == model_name
        assert config["data"]["lookback"] == lookback
        assert config["training"]["backend"] == "torch"
        assert int(config["training"]["epochs"]) > 0
        if model_name == "improved_bnn":
            assert config["training"]["batch_size"] == 64
            assert config["training"]["early_stopping"]["enabled"] is True
            assert int(config["training"]["early_stopping"]["patience"]) > 0
