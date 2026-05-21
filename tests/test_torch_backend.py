import numpy as np

from src.data.pv import WindowArrays
from src.evaluation.predictor import predict_arrays
from src.models.registry import build_model
from src.models.torch_models import BayesianConv1d, BayesianLinear
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


def test_torch_trainer_restores_best_validation_epoch():
    import torch

    class ConstantMeanModel(torch.nn.Module):
        is_torch_model = True

        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            batch_size = len(batch["direct"])
            mean = self.mean.expand(batch_size, 1)
            log_var = torch.zeros_like(mean)
            return mean, log_var

        def kl_loss(self):
            return torch.zeros((), device=self.mean.device)

    train_arrays = WindowArrays(
        history=np.zeros((8, 1, 1), dtype=np.float32),
        weather=np.zeros((8, 1, 1), dtype=np.float32),
        direct=np.zeros((8, 1), dtype=np.float32),
        target=np.full((8, 1), 10.0, dtype=np.float32),
    )
    val_arrays = WindowArrays(
        history=np.zeros((8, 1, 1), dtype=np.float32),
        weather=np.zeros((8, 1, 1), dtype=np.float32),
        direct=np.zeros((8, 1), dtype=np.float32),
        target=np.zeros((8, 1), dtype=np.float32),
    )
    config = {"training": {"device": "auto", "epochs": 4, "batch_size": 8, "lr": 0.2, "weight_decay": 0.0}}
    model = ConstantMeanModel()

    history = train_torch_model(model, train_arrays, config, validation_arrays=val_arrays)

    val_rmses = [item["val_rmse"] for item in history]
    restored = evaluate_torch_model(model, val_arrays, config=config)["rmse"]
    assert restored == min(val_rmses)


def test_evaluate_torch_model_uses_mc_samples_for_stochastic_models():
    import torch

    class CountingStochasticModel(torch.nn.Module):
        is_torch_model = True
        stochastic_predict = True

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            mean = torch.full((len(batch["direct"]), 1), float(self.calls), device=self.anchor.device)
            return mean, torch.zeros_like(mean)

    arrays = WindowArrays(
        history=np.zeros((2, 1, 1), dtype=np.float32),
        weather=np.zeros((2, 1, 1), dtype=np.float32),
        direct=np.zeros((2, 1), dtype=np.float32),
        target=np.full((2, 1), 2.0, dtype=np.float32),
    )
    model = CountingStochasticModel()

    metrics = evaluate_torch_model(model, arrays, config={"evaluation": {"n_samples": 3}})

    assert metrics["mae"] == 0.0
    assert model.calls == 3


def test_improved_bnn_matches_tab3_fixed_structure():
    config = {
        "data": {
            "lookback": 6,
            "horizon": 1,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "improved_bnn", "hidden_dim": 8, "branch_dim": 5, "conv_kernel": 3},
        "training": {"backend": "torch", "device": "auto"},
    }

    model = build_model(config)

    history_layers = [module for module in model.history_fc if isinstance(module, BayesianLinear)]
    fusion_layers = [module for module in model.fusion if isinstance(module, BayesianLinear)]

    assert [(layer.in_features, layer.out_features) for layer in history_layers] == [(6, 32), (32, 64), (64, 16)]
    assert model.history_conv1.out_channels == 32
    assert model.history_conv1.kernel_size == 5
    assert model.history_conv2.out_channels == 32
    assert model.history_conv2.kernel_size == 5
    assert model.conv_pool.kernel_size == (5,)
    assert [(layer.in_features, layer.out_features) for layer in fusion_layers] == [(50, 32), (32, 16)]
    assert not hasattr(model, "weather_branch")
    assert not hasattr(model, "direct_branch")
    assert any(isinstance(module, BayesianLinear) for module in model.modules())
    assert any(isinstance(module, BayesianConv1d) for module in model.modules())
    assert float(model.kl_loss().detach().cpu()) > 0.0


def test_torch_baselines_use_distinct_architectures():
    base_config = {
        "data": {
            "lookback": 4,
            "horizon": 2,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "training": {"backend": "torch", "device": "auto"},
    }

    mlp = build_model({**base_config, "model": {"name": "mlp_baseline", "hidden_dim": 8}})
    cnn = build_model({**base_config, "model": {"name": "cnn_baseline", "hidden_dim": 8, "branch_dim": 4}})
    bnn = build_model({**base_config, "model": {"name": "improved_bnn", "hidden_dim": 8, "branch_dim": 4}})

    assert type(mlp) is not type(bnn)
    assert not any(isinstance(module, (BayesianLinear, BayesianConv1d)) for module in mlp.modules())
    assert not any(module.__class__.__name__ == "Conv1d" for module in mlp.modules())
    assert any(module.__class__.__name__ == "Conv1d" for module in cnn.modules())
    assert not any(isinstance(module, (BayesianLinear, BayesianConv1d)) for module in cnn.modules())


def test_mc_dropout_predict_arrays_uses_multiple_dropout_samples():
    config = {
        "data": {
            "lookback": 3,
            "horizon": 2,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "mc_dropout", "hidden_dim": 8, "branch_dim": 4, "dropout": 0.8},
        "training": {"backend": "torch", "device": "auto"},
        "evaluation": {"n_samples": 6},
    }
    model = build_model(config)
    arrays = WindowArrays(
        history=np.ones((5, 3, 1), dtype=np.float32),
        weather=np.ones((5, 2, 1), dtype=np.float32),
        direct=np.ones((5, 1), dtype=np.float32),
        target=np.zeros((5, 2), dtype=np.float32),
    )

    mean, log_var = predict_arrays(model, arrays, config=config)

    assert mean.shape == (5, 2)
    assert log_var.shape == (5, 2)
    assert np.any(np.exp(log_var) > 1e-6)
