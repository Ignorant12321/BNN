import numpy as np

from src.data.pv import WindowArrays
from src.evaluation.predictor import predict_arrays
from src.models.registry import build_model
from src.models.torch_models import BayesianConv1d, BayesianLinear
from src.training.torch_trainer import (
    arrays_to_torch_dataset,
    early_stopping_monitor_metric,
    evaluate_torch_model,
    resolve_torch_device,
    train_torch_model,
)


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


def test_bayesian_linear_accepts_zero_input_features_without_division_error():
    import torch

    layer = BayesianLinear(0, 3)
    output = layer(torch.zeros((2, 0), dtype=torch.float32))

    assert output.shape == (2, 3)
    assert float(layer.kl_loss().detach().cpu()) > 0.0


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


def test_mlp_baseline_is_plain_deterministic_model():
    import torch

    config = {
        "data": {
            "lookback": 4,
            "horizon": 2,
            "features": {"history": ["h"], "weather": ["w"], "direct": [], "target": "y"},
        },
        "model": {"name": "mlp_baseline", "hidden_dims": [8, 4]},
        "training": {"backend": "torch", "device": "auto"},
    }
    model = build_model(config)
    batch = {
        "history": torch.ones((3, 4, 1), dtype=torch.float32),
        "weather": torch.ones((3, 2, 1), dtype=torch.float32),
        "direct": torch.zeros((3, 0), dtype=torch.float32),
    }

    output = model(batch)

    assert output.shape == (3, 2)
    assert getattr(model, "deterministic_predict", False) is True
    assert not hasattr(model, "log_var_head")


def test_torch_trainer_uses_mse_for_deterministic_models():
    import torch
    from torch import nn

    class DeterministicTorchModel(nn.Module):
        is_torch_model = True
        deterministic_predict = True

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return batch["direct"] * self.weight

    arrays = WindowArrays(
        history=np.zeros((8, 1, 1), dtype=np.float32),
        weather=np.zeros((8, 1, 1), dtype=np.float32),
        direct=np.arange(8, dtype=np.float32).reshape(8, 1),
        target=(3 * np.arange(8, dtype=np.float32)).reshape(8, 1),
    )
    model = DeterministicTorchModel()

    train_torch_model(
        model,
        arrays,
        {"training": {"device": "cpu", "epochs": 30, "batch_size": 8, "lr": 0.01, "weight_decay": 0.0}},
    )

    assert float(model.weight.detach()) > 0.0


def test_predict_arrays_marks_deterministic_torch_uncertainty_as_nan():
    config = {
        "data": {
            "lookback": 2,
            "horizon": 2,
            "features": {"history": ["h"], "weather": ["w"], "direct": [], "target": "y"},
        },
        "model": {"name": "mlp_baseline", "hidden_dims": [8, 4]},
        "training": {"backend": "torch", "device": "auto"},
    }
    arrays = WindowArrays(
        history=np.ones((3, 2, 1), dtype=np.float32),
        weather=np.ones((3, 2, 1), dtype=np.float32),
        direct=np.zeros((3, 0), dtype=np.float32),
        target=np.zeros((3, 2), dtype=np.float32),
    )
    model = build_model(config)

    mean, log_var = predict_arrays(model, arrays, config=config)

    assert mean.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert np.isnan(log_var).all()


def test_torch_trainer_disables_adam_foreach_by_default(monkeypatch):
    import torch
    import src.training.torch_trainer as torch_trainer

    class RecordingAdamW:
        kwargs = None

        def __init__(self, params, **kwargs):
            self.params = list(params)
            RecordingAdamW.kwargs = kwargs

        def zero_grad(self, set_to_none=True):
            for parameter in self.params:
                parameter.grad = None

        def step(self):
            pass

    class SimpleTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            mean = torch.zeros_like(batch["target"]) + self.bias
            return mean, torch.zeros_like(mean)

        def kl_loss(self):
            return self.bias * 0.0

    monkeypatch.setattr(torch_trainer.torch.optim, "AdamW", RecordingAdamW)
    arrays = WindowArrays(
        history=np.zeros((2, 1, 1), dtype=np.float32),
        weather=np.zeros((2, 1, 1), dtype=np.float32),
        direct=np.zeros((2, 1), dtype=np.float32),
        target=np.ones((2, 1), dtype=np.float32),
    )

    train_torch_model(
        SimpleTorchModel(),
        arrays,
        {"training": {"device": "cpu", "epochs": 1, "batch_size": 2, "lr": 0.01}},
    )

    assert RecordingAdamW.kwargs["foreach"] is False


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


def test_torch_trainer_stops_early_after_validation_patience():
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
    config = {
        "training": {
            "device": "auto",
            "epochs": 20,
            "batch_size": 8,
            "lr": 0.2,
            "weight_decay": 0.0,
            "early_stopping": {"enabled": True, "patience": 1, "min_delta": 0.0},
        }
    }
    model = ConstantMeanModel()

    history = train_torch_model(model, train_arrays, config, validation_arrays=val_arrays)

    assert len(history) < 20
    assert history[-1]["early_stop"] == 1.0


def test_torch_training_records_validation_loss():
    import torch

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            template = batch.get("target", batch["direct"])
            if not isinstance(template, torch.Tensor):
                template = torch.as_tensor(template, device=self.bias.device)
            bias = self.bias if "target" in batch else self.bias.detach()
            mean = bias.expand_as(template)
            log_var = torch.zeros_like(mean)
            return mean, log_var

        def kl_loss(self):
            return torch.zeros((), device=self.bias.device)

    arrays = WindowArrays(
        history=np.zeros((2, 1, 1), dtype=np.float32),
        weather=np.zeros((2, 1, 1), dtype=np.float32),
        direct=np.zeros((2, 1), dtype=np.float32),
        target=np.zeros((2, 1), dtype=np.float32),
    )
    config = {"training": {"device": "auto", "epochs": 1, "batch_size": 2, "lr": 0.0, "kl_beta": 0.0}}

    history = train_torch_model(ZeroModel(), arrays, config, validation_arrays=arrays)

    assert "val_loss" in history[0]
    assert history[0]["val_loss"] == 0.0


def test_early_stopping_monitor_metric_can_use_generation_validation_metric():
    training = {"early_stopping": {"enabled": True, "metric": "val_generation_nrmse"}}

    assert early_stopping_monitor_metric(training) == "val_generation_nrmse"
    assert early_stopping_monitor_metric({}) == "val_rmse"


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


def test_predict_arrays_samples_recursive_trajectory_outputs():
    import torch

    class RecursiveGaussianModel(torch.nn.Module):
        is_torch_model = True
        stochastic_predict = True
        sample_recursive_trajectory = True

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.horizon = 2
            self.observed_roll_values = []

        def forward_step(self, rolling_history, weather_step, previous_power):
            self.observed_roll_values.append(float(previous_power.detach().cpu()[0, 0]))
            mean = previous_power + 1.0
            log_var = torch.zeros_like(mean)
            return mean, log_var

    arrays = WindowArrays(
        history=np.zeros((1, 2, 1), dtype=np.float32),
        weather=np.zeros((1, 2, 1), dtype=np.float32),
        direct=np.zeros((1, 1), dtype=np.float32),
        target=np.zeros((1, 2), dtype=np.float32),
    )
    model = RecursiveGaussianModel()

    torch.manual_seed(0)
    mean, log_var = predict_arrays(model, arrays, config={"evaluation": {"n_samples": 2}})

    assert mean.shape == (1, 2)
    assert log_var.shape == (1, 2)
    assert model.observed_roll_values[0] == 0.0
    assert model.observed_roll_values[2] == 0.0
    assert model.observed_roll_values[1] != 1.0
    assert model.observed_roll_values[3] != 1.0


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


def test_improved_bnn_supports_zero_lookback_with_weather_and_direct_only():
    config = {
        "data": {
            "lookback": 0,
            "horizon": 2,
            "features": {"history": [], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "improved_bnn"},
        "training": {"backend": "torch", "device": "auto"},
        "evaluation": {"n_samples": 2},
    }
    model = build_model(config)
    arrays = WindowArrays(
        history=np.zeros((3, 0, 0), dtype=np.float32),
        weather=np.ones((3, 2, 1), dtype=np.float32),
        direct=np.ones((3, 1), dtype=np.float32),
        target=np.zeros((3, 2), dtype=np.float32),
    )

    mean, log_var = predict_arrays(model, arrays, config=config)

    assert mean.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert not hasattr(model, "history_fc")
    assert not hasattr(model, "history_conv1")
    assert float(model.kl_loss().detach().cpu()) > 0.0


def test_pv_usibnn_uses_ultra_short_term_branches_and_mc_sampling():
    config = {
        "data": {
            "lookback": 16,
            "horizon": 16,
            "features": {
                "history": ["AC_POWER"],
                "weather": [
                    "IRRADIATION",
                    "AMBIENT_TEMPERATURE",
                    "MODULE_TEMPERATURE",
                    "hour_sin",
                    "hour_cos",
                    "dayofyear_sin",
                    "dayofyear_cos",
                    "is_generation_time",
                ],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "model": {"name": "pv_usibnn"},
        "training": {"backend": "torch", "device": "auto"},
        "evaluation": {"n_samples": 3},
    }
    model = build_model(config)
    arrays = WindowArrays(
        history=np.ones((4, 16, 1), dtype=np.float32),
        weather=np.ones((4, 16, 8), dtype=np.float32),
        direct=np.ones((4, 1), dtype=np.float32),
        target=np.zeros((4, 16), dtype=np.float32),
    )

    mean, log_var = predict_arrays(model, arrays, config=config)

    assert model.stochastic_predict is True
    assert model.solar_time_indices == (0, 3, 4, 5, 6, 7)
    assert model.weather_indices == (1, 2)
    assert not hasattr(model, "history_fc")
    assert not hasattr(model, "direct_branch")
    assert model.history_conv1.kernel_size == 3
    assert model.history_conv1.out_channels == 32
    assert model.history_conv2.kernel_size == 3
    assert model.history_conv2.out_channels == 32
    assert model.history_pool.output_size == 4
    assert not hasattr(model, "conv_pool")
    assert model.history_projection.in_features == 128
    assert model.fusion[0].in_features == 65
    assert mean.shape == (4, 16)
    assert log_var.shape == (4, 16)
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
