"""模型注册表。

功能：
    根据配置中的 `model.name` 构造模型，集中管理可用模型名称。

使用：
    单模型训练：
    python -m src.experiments.train --config configs/models/bnn_24h.yaml

    多模型对比：
    python -m src.experiments.compare_results --config configs/compare/main.yaml
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.data.pv import feature_dimensions_from_config
from src.models.baselines import FlattenedMLPBaseline, HistoryCNNBaseline, MCDropoutBaseline
from src.models.improved_bnn import ImprovedBayesianPVNet


ModelBuilder = Callable[[dict[str, Any]], object]


def _horizon(config: dict[str, Any]) -> int:
    return int(config["data"]["horizon"])


def _torch_backend(config: dict[str, Any]) -> bool:
    return str(config.get("training", {}).get("backend", "numpy")).lower() == "torch"


def _torch_model(config: dict[str, Any], feature_mode: str, dropout: float = 0.0):
    from src.models.torch_models import TorchPVNet

    data = config["data"]
    model_config = config.get("model", {})
    history_features, weather_features, direct_features = feature_dimensions_from_config(config)
    return TorchPVNet(
        lookback=int(data["lookback"]),
        horizon=int(data["horizon"]),
        history_features=history_features,
        weather_features=weather_features,
        direct_features=direct_features,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        branch_dim=int(model_config.get("branch_dim", 64)),
        feature_mode=feature_mode,
        dropout=dropout,
    )


def _build_improved_bnn(config: dict[str, Any]) -> ImprovedBayesianPVNet:
    if _torch_backend(config):
        return _torch_model(config, feature_mode="all")
    return ImprovedBayesianPVNet(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


def _build_mlp_baseline(config: dict[str, Any]) -> FlattenedMLPBaseline:
    if _torch_backend(config):
        return _torch_model(config, feature_mode="all")
    return FlattenedMLPBaseline(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


def _build_cnn_baseline(config: dict[str, Any]) -> HistoryCNNBaseline:
    if _torch_backend(config):
        return _torch_model(config, feature_mode="history")
    return HistoryCNNBaseline(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


def _build_mc_dropout(config: dict[str, Any]) -> MCDropoutBaseline:
    if _torch_backend(config):
        return _torch_model(config, feature_mode="history_weather", dropout=float(config.get("model", {}).get("dropout", 0.2)))
    return MCDropoutBaseline(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "improved_bnn": _build_improved_bnn,
    "mlp_baseline": _build_mlp_baseline,
    "cnn_baseline": _build_cnn_baseline,
    "mc_dropout": _build_mc_dropout,
}


def supported_model_names() -> tuple[str, ...]:
    """返回 YAML 配置中可以使用的模型名。"""
    return tuple(MODEL_REGISTRY)


def build_model(config: dict[str, Any]):
    """根据 `config['model']['name']` 构造模型。"""
    model_name = config.get("model", {}).get("name", "improved_bnn")
    try:
        builder = MODEL_REGISTRY[model_name]
    except KeyError as error:
        supported = ", ".join(supported_model_names())
        raise ValueError(f"unsupported model.name {model_name!r}; supported values: {supported}") from error
    return builder(config)
