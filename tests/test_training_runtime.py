"""训练运行时配置测试。"""

from __future__ import annotations

from types import SimpleNamespace

from src.train import _amp_enabled, _build_loader_kwargs


def test_build_loader_kwargs_enables_cuda_input_pipeline_options() -> None:
    """CUDA 训练时应能通过配置启用更快的数据搬运选项。"""
    config = {
        "training": {
            "batch_size": 128,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
        }
    }

    kwargs = _build_loader_kwargs(config, SimpleNamespace(type="cuda"), shuffle=True, platform="linux")

    assert kwargs["batch_size"] == 128
    assert kwargs["shuffle"] is True
    assert kwargs["num_workers"] == 2
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True


def test_build_loader_kwargs_disables_workers_on_windows_cuda_by_default() -> None:
    """Windows CUDA 训练默认不启动 DataLoader 子进程，避免重复加载 CUDA DLL。"""
    config = {
        "training": {
            "batch_size": 128,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
        }
    }

    kwargs = _build_loader_kwargs(config, SimpleNamespace(type="cuda"), shuffle=True, platform="win32")

    assert kwargs["num_workers"] == 0
    assert kwargs["pin_memory"] is True
    assert "persistent_workers" not in kwargs


def test_build_loader_kwargs_omits_persistent_workers_without_workers() -> None:
    """num_workers=0 时不能传 persistent_workers=True。"""
    config = {
        "training": {
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": True,
        }
    }

    kwargs = _build_loader_kwargs(config, SimpleNamespace(type="cpu"), shuffle=False)

    assert kwargs["batch_size"] == 64
    assert kwargs["shuffle"] is False
    assert kwargs["num_workers"] == 0
    assert kwargs["pin_memory"] is False
    assert "persistent_workers" not in kwargs


def test_amp_enabled_only_on_cuda_when_configured() -> None:
    """混合精度只应在 CUDA 设备且配置开启时启用。"""
    assert _amp_enabled({"training": {"amp": True}}, SimpleNamespace(type="cuda")) is True
    assert _amp_enabled({"training": {"amp": True}}, SimpleNamespace(type="cpu")) is False
    assert _amp_enabled({"training": {"amp": False}}, SimpleNamespace(type="cuda")) is False
