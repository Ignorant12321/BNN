"""训练运行时配置测试。"""

from __future__ import annotations

from types import SimpleNamespace

from src.train import _amp_enabled, _build_loader_kwargs, _build_training_summary_lines, _color_training_log_message


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
    assert "validation_metrics: metrics/validation_metrics.json\n  metric    value\n  rmse      10.123456" in text
    assert "  mae       5.500000" in text
    assert "  picp_90   0.910000" in text
    assert "test_metrics: metrics/metrics.json\n  metric    value\n  rmse      11.200000" in text
    assert "  mae       6.000000" in text
    assert "  picp_90   0.880000" in text


def test_color_training_log_message_highlights_epochs_losses_and_options() -> None:
    """训练控制台日志应克制地突出 epoch、loss 和运行参数。"""
    epoch_text = _color_training_log_message("epoch=003 train_loss=1.234567 val_loss=0.987654")
    options_text = _color_training_log_message("Runtime options: batch_size=128, num_workers=0, pin_memory=False, amp=True")
    summary_text = _color_training_log_message(
        "completed_epochs=12/20 best_epoch=9 best_val_loss=0.123457 early_stopping_epoch=none"
    )

    assert "\033[32mepoch=003\033[0m" in epoch_text
    assert "\033[35mtrain_loss=1.234567\033[0m" in epoch_text
    assert "\033[35mval_loss=0.987654\033[0m" in epoch_text
    assert "\033[34mbatch_size=128\033[0m" in options_text
    assert "\033[34mamp=True\033[0m" in options_text
    assert "\033[32mcompleted_epochs=12/20\033[0m" in summary_text
    assert "\033[35mbest_val_loss=0.123457\033[0m" in summary_text
    assert "\033[31m" not in epoch_text + options_text + summary_text


def test_color_training_log_message_leaves_plain_logs_uncolored() -> None:
    """设备和 run_dir 等普通日志保持默认控制台颜色。"""
    text = _color_training_log_message("Device status: device=cpu, cuda_available=False")

    assert text == "Device status: device=cpu, cuda_available=False"
