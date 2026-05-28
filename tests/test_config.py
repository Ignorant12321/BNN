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
        "configs/models/bnn/0h.yaml": ("improved_bnn", 0),
        "configs/models/bnn/1h.yaml": ("improved_bnn", 4),
        "configs/models/bnn/4h.yaml": ("improved_bnn", 16),
        "configs/models/bnn/recursive_4h.yaml": ("improved_bnn", 16),
        "configs/models/bnn/8h.yaml": ("improved_bnn", 32),
        "configs/models/bnn/12h.yaml": ("improved_bnn", 48),
        "configs/models/bnn/24h.yaml": ("improved_bnn", 96),
        "configs/models/mlp/24h.yaml": ("mlp_baseline", 96),
        "configs/models/mlp/plain_4h.yaml": ("mlp_baseline", 16),
        "configs/models/mlp/recursive_4h.yaml": ("mlp_baseline", 16),
        "configs/models/cnn/recursive_4h.yaml": ("cnn_baseline", 16),
        "configs/models/cnn/24h.yaml": ("cnn_baseline", 96),
        "configs/models/lstm/recursive_4h.yaml": ("lstm_baseline", 16),
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
            if config_path == "configs/models/bnn/4h.yaml":
                assert config["training"]["early_stopping"]["metric"] == "val_generation_nrmse"


def test_bnn_recursive_four_hour_config_marks_recursive_strategy():
    config = load_config("configs/models/bnn/recursive_4h.yaml")

    assert config["model"]["name"] == "improved_bnn"
    assert config["data"]["lookback"] == 16
    assert config["data"]["horizon"] == 16
    assert config["strategy"]["name"] == "recursive"
    assert config["strategy"]["train_horizon"] == 1
    assert config["strategy"]["forecast_horizon"] == 16


def test_bnn_configs_use_irradiation_without_explicit_time_features():
    four_hour = load_config("configs/models/bnn/4h.yaml")
    day = load_config("configs/models/bnn/24h.yaml")
    expected_weather = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
    ]

    assert four_hour["data"]["features"]["weather"] == expected_weather
    assert day["data"]["features"]["weather"] == expected_weather


def test_pv_usibnn_config_defaults_to_four_hour_ultra_short_term_inputs():
    config = load_config("configs/models/bnn/pv_usibnn.yaml")

    assert config["model"]["name"] == "pv_usibnn"
    assert config["data"]["lookback"] == 16
    assert config["data"]["horizon"] == 16
    assert config["data"]["features"]["history"] == ["AC_POWER"]
    assert config["data"]["features"]["weather"] == [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_generation_time",
    ]
    assert config["data"]["features"]["direct"] == ["AC_POWER"]


def test_plain_mlp_four_hour_config_uses_weather_time_inputs_without_direct_power():
    config = load_config("configs/models/mlp/plain_4h.yaml")

    assert config["model"]["name"] == "mlp_baseline"
    assert config["model"]["hidden_dims"] == [128, 64]
    assert config["data"]["lookback"] == 16
    assert config["data"]["horizon"] == 16
    assert config["data"]["features"]["history"] == ["AC_POWER"]
    assert config["data"]["features"]["weather"] == [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_generation_time",
    ]
    assert config["data"]["features"]["direct"] == []
    assert config["training"]["early_stopping"]["metric"] == "val_generation_nrmse"


def test_recursive_point_baseline_configs_use_same_direct_inputs():
    for config_path in (
        "configs/models/mlp/recursive_4h.yaml",
        "configs/models/cnn/recursive_4h.yaml",
        "configs/models/lstm/recursive_4h.yaml",
    ):
        config = load_config(config_path)

        assert config["data"]["lookback"] == 16
        assert config["data"]["horizon"] == 16
        assert config["data"]["features"]["history"] == ["AC_POWER"]
        assert config["data"]["features"]["weather"] == [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
        ]
        assert config["data"]["features"]["direct"] == ["AC_POWER"]
