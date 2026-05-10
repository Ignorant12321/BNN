"""训练运行时配置测试。"""

from __future__ import annotations

from types import SimpleNamespace

from src.train import _amp_enabled, _build_loader_kwargs, _build_training_summary_lines


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


def test_build_training_summary_lines_includes_duration_status_and_metrics(tmp_path) -> None:
    """训练日志结尾应包含耗时、训练总结和核心评估指标。"""
    val_result = SimpleNamespace(
        split="val",
        metrics={"rmse": 10.123456, "mae": 5.5, "picp_90": 0.91},
        outputs=SimpleNamespace(metrics=tmp_path / "metrics" / "validation_metrics.json"),
    )
    test_result = SimpleNamespace(
        split="test",
        metrics={"rmse": 11.2, "mae": 6.0, "picp_90": 0.88},
        outputs=SimpleNamespace(metrics=tmp_path / "metrics" / "metrics.json"),
    )

    lines = _build_training_summary_lines(
        run_dir=tmp_path,
        elapsed_seconds=3723.4,
        completed_epochs=12,
        requested_epochs=20,
        best_epoch=9,
        best_val=0.1234567,
        early_stop_epoch=12,
        train_losses=[0.9, 0.4],
        val_losses=[0.8, 0.5],
        evaluation_results=[val_result, test_result],
    )

    text = "\n".join(lines)
    assert "Training Summary" in text
    assert "elapsed_time=01:02:03" in text
    assert "completed_epochs=12/20" in text
    assert "best_epoch=9" in text
    assert "best_val_loss=0.123457" in text
    assert "early_stopping_epoch=12" in text
    assert "final_train_loss=0.400000" in text
    assert "validation_metrics=metrics/validation_metrics.json rmse=10.123456 mae=5.500000 picp_90=0.910000" in text
    assert "test_metrics=metrics/metrics.json rmse=11.200000 mae=6.000000 picp_90=0.880000" in text
