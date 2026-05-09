from tests.conftest import torch_required


@torch_required
def test_improved_bnn_forward_shapes_and_kl():
    import torch

    from src.models.improved_bnn import ImprovedBayesianPVNet

    model = ImprovedBayesianPVNet(
        history_features=5,
        weather_features=3,
        time_features=6,
        direct_features=3,
        horizon=16,
        hidden_dim=32,
    )
    batch = {
        "history": torch.randn(4, 96, 5),
        "weather": torch.randn(4, 16, 3),
        "time": torch.randn(4, 16, 6),
        "direct": torch.randn(4, 3),
    }

    mean, log_var = model(batch)

    assert mean.shape == (4, 16)
    assert log_var.shape == (4, 16)
    assert model.kl_loss().ndim == 0
