"""pytest 共享配置。

当前环境可能没有安装 PyTorch，因此模型相关测试通过 `torch_required`
标记按需跳过，避免影响纯数据处理测试。
"""

import importlib.util

import pytest


def has_torch() -> bool:
    """检查当前 Python 环境是否安装了 torch。"""
    return importlib.util.find_spec("torch") is not None


# 供测试文件复用的跳过标记。
torch_required = pytest.mark.skipif(
    not has_torch(),
    reason="torch is not installed in this environment",
)
