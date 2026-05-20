"""运行环境检测命令。

功能：
    检查当前 Python 环境是否可以导入 PyTorch，以及 CUDA 是否可用。

使用：
    python -m src.environment
"""

from __future__ import annotations

import json
import subprocess
import sys


def describe_torch_environment() -> dict[str, object]:
    """返回 PyTorch/CUDA 环境状态。

    这里使用子进程导入 torch。某些 Windows/Conda 环境的 OpenMP DLL 冲突会
    在导入阶段直接终止进程，子进程检测可以避免主命令一起崩溃。
    """
    probe = """
import json
import torch
cuda = bool(torch.cuda.is_available())
print(json.dumps({
    "available": True,
    "version": torch.__version__,
    "cuda_available": cuda,
    "device": "cuda" if cuda else "cpu",
    "cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
}))
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "available": False,
            "cuda_available": False,
            "device": "cpu",
            "error": (result.stderr or result.stdout).strip(),
        }
    return json.loads(result.stdout)


def main() -> None:
    """命令行入口。"""
    status = describe_torch_environment()
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
