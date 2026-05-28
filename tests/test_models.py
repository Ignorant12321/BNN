import numpy as np
import pytest

from src.models.registry import build_model, supported_model_names


def test_supported_model_names_are_stable():
    assert supported_model_names() == (
        "improved_bnn",
        "pv_usibnn",
        "pv_usibnn_recursive",
        "mlp_baseline",
        "cnn_baseline",
        "lstm_baseline",
        "mc_dropout",
    )


def test_build_model_returns_probabilistic_interface():
    config = {
        "data": {
            "lookback": 4,
            "horizon": 3,
            "features": {
                "history": ["h1", "h2"],
                "weather": ["w1", "w2", "w3"],
                "direct": ["d1"],
                "target": "y",
            },
        },
        "training": {"backend": "numpy"},
        "model": {"name": "mlp_baseline", "hidden_dim": 8},
    }
    batch = {
        "history": np.ones((5, 4, 2), dtype=np.float32),
        "weather": np.ones((5, 3, 3), dtype=np.float32),
        "direct": np.ones((5, 1), dtype=np.float32),
    }

    model = build_model(config)
    mean, log_var = model(batch)

    assert mean.shape == (5, 3)
    assert log_var.shape == (5, 3)
    assert np.isscalar(model.kl_loss())


def test_baseline_fit_learns_direct_signal():
    config = {
        "data": {
            "lookback": 2,
            "horizon": 2,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "training": {"backend": "numpy"},
        "model": {"name": "mlp_baseline", "ridge_alpha": 0.0},
    }
    model = build_model(config)
    arrays = type(
        "Arrays",
        (),
        {
            "history": np.array([[[0.0], [1.0]], [[0.0], [2.0]], [[0.0], [3.0]]], dtype=np.float32),
            "weather": np.zeros((3, 2, 1), dtype=np.float32),
            "direct": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "target": np.array([[2.0, 3.0], [4.0, 6.0], [6.0, 9.0]], dtype=np.float32),
        },
    )()

    before, _ = model({"history": arrays.history, "weather": arrays.weather, "direct": arrays.direct})
    model.fit(arrays)
    after, _ = model({"history": arrays.history, "weather": arrays.weather, "direct": arrays.direct})

    assert np.mean((after - arrays.target) ** 2) < np.mean((before - arrays.target) ** 2)
    np.testing.assert_allclose(after, arrays.target, atol=1e-4)


def test_build_model_rejects_unknown_name():
    config = {
        "data": {
            "lookback": 4,
            "horizon": 3,
            "features": {"history": ["h"], "weather": ["w"], "direct": ["d"], "target": "y"},
        },
        "model": {"name": "unknown"},
    }

    with pytest.raises(ValueError, match="unsupported model.name"):
        build_model(config)


def test_recursive_pv_usibnn_rolls_predictions_across_horizon():
    torch = pytest.importorskip("torch")
    from src.models.pv_usibnn_recursive import PVRecursiveUltraShortTermIBNN

    model = PVRecursiveUltraShortTermIBNN(
        lookback=4,
        horizon=3,
        history_features=1,
        weather_feature_names=[
            "IRRADIATION",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "hour_sin",
            "hour_cos",
            "dayofyear_sin",
            "dayofyear_cos",
            "is_generation_time",
        ],
        direct_features=1,
        teacher_forcing_ratio=1.0,
    )
    batch = {
        "history": torch.ones((2, 4, 1), dtype=torch.float32),
        "weather": torch.ones((2, 3, 8), dtype=torch.float32),
        "direct": torch.ones((2, 1), dtype=torch.float32),
        "target": torch.ones((2, 3), dtype=torch.float32),
    }

    mean, log_var = model(batch)

    assert mean.shape == (2, 3)
    assert log_var.shape == (2, 3)


def test_build_model_returns_recursive_pv_usibnn():
    pytest.importorskip("torch")
    from src.models.pv_usibnn_recursive import PVRecursiveUltraShortTermIBNN

    config = {
        "data": {
            "lookback": 4,
            "horizon": 3,
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
        "training": {"backend": "torch"},
        "model": {"name": "pv_usibnn_recursive", "teacher_forcing_ratio": 0.5},
    }

    model = build_model(config)

    assert isinstance(model, PVRecursiveUltraShortTermIBNN)


def test_torch_lstm_baseline_is_deterministic_one_step_model():
    torch = pytest.importorskip("torch")

    config = {
        "data": {
            "lookback": 4,
            "horizon": 1,
            "features": {
                "history": ["AC_POWER"],
                "weather": ["IRRADIATION", "AMBIENT_TEMPERATURE"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            },
        },
        "training": {"backend": "torch"},
        "model": {"name": "lstm_baseline", "hidden_dim": 8},
    }

    model = build_model(config)
    batch = {
        "history": torch.ones((3, 4, 1), dtype=torch.float32),
        "weather": torch.ones((3, 1, 2), dtype=torch.float32),
        "direct": torch.ones((3, 1), dtype=torch.float32),
    }

    output = model(batch)

    assert getattr(model, "deterministic_predict", False) is True
    assert output.shape == (3, 1)
