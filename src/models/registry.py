"""模型注册表。

功能：
    根据配置中的 `model.name` 构造模型，集中管理可用模型名称。

使用：
    单模型训练：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml

    多模型对比：
    python -m src.experiments.compare --config configs/compare/main.yaml
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.data.pv import feature_columns_from_config, feature_dimensions_from_config
from src.models.baselines import FlattenedMLPBaseline, HistoryCNNBaseline, MCDropoutBaseline
from src.models.improved_bnn import ImprovedBayesianPVNet


ModelBuilder = Callable[[dict[str, Any]], object]


def _horizon(config: dict[str, Any]) -> int:
    return int(config["data"]["horizon"])


def _torch_backend(config: dict[str, Any]) -> bool:
    return str(config.get("training", {}).get("backend", "numpy")).lower() == "torch"


def _model_dimensions(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, int, int]:
    data = config["data"]
    model_config = config.get("model", {})
    history_features, weather_features, direct_features = feature_dimensions_from_config(config)
    return data, model_config, history_features, weather_features, direct_features


def _build_improved_bnn(config: dict[str, Any]) -> ImprovedBayesianPVNet:
    if _torch_backend(config):
        from src.models.torch_models import ImprovedBayesianTorchNet

        data, model_config, history_features, weather_features, direct_features = _model_dimensions(config)
        return ImprovedBayesianTorchNet(
            lookback=int(data["lookback"]),
            horizon=int(data["horizon"]),
            history_features=history_features,
            weather_features=weather_features,
            direct_features=direct_features,
        )
    return ImprovedBayesianPVNet(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


def _build_pv_usibnn(config: dict[str, Any]):
    if not _torch_backend(config):
        raise ValueError("pv_usibnn requires training.backend: torch")
    from src.models.pv_usibnn import PVUltraShortTermIBNN

    data = config["data"]
    columns = feature_columns_from_config(config)
    return PVUltraShortTermIBNN(
        lookback=int(data["lookback"]),
        horizon=int(data["horizon"]),
        history_features=len(columns.history),
        weather_feature_names=columns.weather,
        direct_features=len(columns.direct),
    )


def _build_pv_usibnn_recursive(config: dict[str, Any]):
    if not _torch_backend(config):
        raise ValueError("pv_usibnn_recursive requires training.backend: torch")
    from src.models.pv_usibnn_recursive import PVRecursiveUltraShortTermIBNN

    data = config["data"]
    model_config = config.get("model", {})
    columns = feature_columns_from_config(config)
    return PVRecursiveUltraShortTermIBNN(
        lookback=int(data["lookback"]),
        horizon=int(data["horizon"]),
        history_features=len(columns.history),
        weather_feature_names=columns.weather,
        direct_features=len(columns.direct),
        hidden_dim=int(model_config.get("hidden_dim", 32)),
        context_dim=int(model_config.get("context_dim", 32)),
        teacher_forcing_ratio=float(model_config.get("teacher_forcing_ratio", 0.0)),
    )


def _build_mlp_baseline(config: dict[str, Any]) -> FlattenedMLPBaseline:
    if _torch_backend(config):
        from src.models.torch_models import MLPBaselineTorchNet

        data, model_config, history_features, weather_features, direct_features = _model_dimensions(config)
        return MLPBaselineTorchNet(
            lookback=int(data["lookback"]),
            horizon=int(data["horizon"]),
            history_features=history_features,
            weather_features=weather_features,
            direct_features=direct_features,
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            hidden_dims=model_config.get("hidden_dims"),
        )
    return FlattenedMLPBaseline(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


def _build_cnn_baseline(config: dict[str, Any]) -> HistoryCNNBaseline:
    if _torch_backend(config):
        from src.models.torch_models import CNNBaselineTorchNet

        data, model_config, history_features, weather_features, direct_features = _model_dimensions(config)
        return CNNBaselineTorchNet(
            lookback=int(data["lookback"]),
            horizon=int(data["horizon"]),
            history_features=history_features,
            weather_features=weather_features,
            direct_features=direct_features,
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            branch_dim=int(model_config.get("branch_dim", 64)),
            conv_kernel=int(model_config.get("conv_kernel", 5)),
        )
    return HistoryCNNBaseline(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


def _build_lstm_baseline(config: dict[str, Any]):
    if not _torch_backend(config):
        raise ValueError("lstm_baseline requires training.backend: torch")
    from src.models.torch_models import LSTMBaselineTorchNet

    data, model_config, history_features, weather_features, direct_features = _model_dimensions(config)
    return LSTMBaselineTorchNet(
        horizon=int(data["horizon"]),
        history_features=history_features,
        weather_features=weather_features,
        direct_features=direct_features,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_layers=int(model_config.get("num_layers", 1)),
    )


def _build_mc_dropout(config: dict[str, Any]) -> MCDropoutBaseline:
    if _torch_backend(config):
        from src.models.torch_models import MCDropoutTorchNet

        data, model_config, history_features, weather_features, _direct_features = _model_dimensions(config)
        return MCDropoutTorchNet(
            lookback=int(data["lookback"]),
            horizon=int(data["horizon"]),
            history_features=history_features,
            weather_features=weather_features,
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            branch_dim=int(model_config.get("branch_dim", 64)),
            dropout=float(model_config.get("dropout", 0.2)),
        )
    return MCDropoutBaseline(horizon=_horizon(config), alpha=float(config.get("model", {}).get("ridge_alpha", 1e-3)))


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "improved_bnn": _build_improved_bnn,
    "pv_usibnn": _build_pv_usibnn,
    "pv_usibnn_recursive": _build_pv_usibnn_recursive,
    "mlp_baseline": _build_mlp_baseline,
    "cnn_baseline": _build_cnn_baseline,
    "lstm_baseline": _build_lstm_baseline,
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
