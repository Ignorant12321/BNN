"""PyTorch 导入与运行时辅助。

功能：
    在当前 Windows/Conda 环境中，导入 PyTorch 前需要允许重复 OpenMP 运行时，
    否则可能出现 libiomp5md.dll 冲突。本模块集中处理这个兼容设置。

使用：
    from src.torch_runtime import import_torch
"""

from __future__ import annotations

import os


def import_torch():
    """设置必要环境变量后导入并返回 torch 模块。"""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    return torch

