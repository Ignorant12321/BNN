"""损失函数测试。"""

from tests.conftest import torch_required


@torch_required
def test_gaussian_nll_is_differentiable():
    """Gaussian NLL 必须能对 mean 和 log_var 反向传播。"""
    import torch

    from src.losses import gaussian_nll

    mean = torch.zeros(2, 3, requires_grad=True)
    log_var = torch.zeros(2, 3, requires_grad=True)
    target = torch.ones(2, 3)

    loss = gaussian_nll(mean, log_var, target)
    loss.backward()

    assert mean.grad is not None
    assert log_var.grad is not None
