"""主模型结构测试。"""

from tests.conftest import torch_required


@torch_required
def test_improved_bnn_forward_shapes_and_kl():
    """Improved BNN 前向输出应为 [batch, horizon]，KL 应为标量。"""
    import torch

    from src.models.improved_bnn import ImprovedBayesianPVNet

    model = ImprovedBayesianPVNet(
        history_features=1,
        weather_features=4,
        direct_features=1,
        horizon=16,
        hidden_dim=32,
    )
    batch = {
        "history": torch.randn(4, 16, 1),
        "weather": torch.randn(4, 16, 4),
        "direct": torch.randn(4, 1),
    }

    mean, log_var = model(batch)

    assert mean.shape == (4, 16)
    assert log_var.shape == (4, 16)
    assert model.kl_loss().ndim == 0
