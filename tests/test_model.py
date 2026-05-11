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


@torch_required
def test_build_model_supports_compare_baselines():
    """对比实验中的模型名都应能构造并输出概率预测接口。"""
    import torch

    from src.evaluation_pipeline import build_model
    from src.features import split_feature_columns

    columns = split_feature_columns()
    base_config = {
        "data": {"lookback": 16, "horizon": 16},
        "model": {"hidden_dim": 32, "branch_dim": 16, "prior_sigma": 1.0},
    }
    batch = {
        "history": torch.randn(3, 16, len(columns.history)),
        "weather": torch.randn(3, 16, len(columns.weather)),
        "direct": torch.randn(3, len(columns.direct)),
    }

    for model_name in ["improved_bnn", "mlp_baseline", "cnn_baseline", "mc_dropout"]:
        config = {**base_config, "model": {**base_config["model"], "name": model_name}}
        model = build_model(config, columns, torch.device("cpu"))

        mean, log_var = model(batch)

        assert mean.shape == (3, 16)
        assert log_var.shape == (3, 16)
        assert model.kl_loss().ndim == 0
