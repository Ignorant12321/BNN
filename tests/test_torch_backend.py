import numpy as np

from src.data.pv import WindowArrays
from src.models.registry import build_model
from src.training.torch_trainer import arrays_to_torch_dataset, evaluate_torch_model, resolve_torch_device, train_torch_model


def test_resolve_torch_device_accepts_auto_or_cuda():
    auto_device = resolve_torch_device("auto")

    assert auto_device.type in {"cpu", "cuda"}


def test_registry_builds_torch_model_when_backend_is_torch():
    config = {
        "data": {
            "lookback": 2,
            "horizon": 1,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "improved_bnn", "hidden_dim": 8},
        "training": {"backend": "torch", "device": "auto"},
    }

    model = build_model(config)

    assert getattr(model, "is_torch_model", False)


def test_torch_trainer_reduces_simple_direct_error():
    arrays = WindowArrays(
        history=np.zeros((16, 2, 1), dtype=np.float32),
        weather=np.zeros((16, 1, 1), dtype=np.float32),
        direct=np.arange(16, dtype=np.float32).reshape(16, 1),
        target=(2 * np.arange(16, dtype=np.float32)).reshape(16, 1),
    )
    config = {
        "data": {
            "lookback": 2,
            "horizon": 1,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "mlp_baseline", "hidden_dim": 16},
        "training": {"backend": "torch", "device": "auto", "epochs": 20, "batch_size": 8, "lr": 0.01, "weight_decay": 0.0},
    }
    model = build_model(config)
    before = evaluate_torch_model(model, arrays, resolve_torch_device("auto"))["rmse"]

    train_torch_model(model, arrays, config)
    after = evaluate_torch_model(model, arrays, resolve_torch_device("auto"))["rmse"]

    assert after < before
    assert len(arrays_to_torch_dataset(arrays)) == 16


def test_improved_bnn_uses_branch_dim_in_torch_backend():
    config = {
        "data": {
            "lookback": 2,
            "horizon": 1,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "improved_bnn", "hidden_dim": 8, "branch_dim": 5},
        "training": {"backend": "torch", "device": "auto"},
    }

    model = build_model(config)

    assert model.history_branch[-2].out_features == 5
    assert model.weather_branch[-2].out_features == 5
