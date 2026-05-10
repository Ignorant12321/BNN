"""通用工具函数测试。"""

from src.utils import create_run_dir, describe_device


def test_describe_device_reports_cuda_name_when_available():
    """训练日志应明确显示 CUDA 是否可用及显卡名称。"""
    text = describe_device("cuda", cuda_available=True, cuda_name="RTX Test")

    assert "device=cuda" in text
    assert "cuda_available=True" in text
    assert "gpu=RTX Test" in text


def test_describe_device_reports_cpu_fallback():
    """CUDA 不可用时日志应说明实际使用 CPU。"""
    text = describe_device("cpu", cuda_available=False, cuda_name=None)

    assert "device=cpu" in text
    assert "cuda_available=False" in text


def test_create_run_dir_writes_default_timestamp_note(tmp_path):
    """实验目录应包含备注文件，默认内容为时间戳目录名。"""
    run_dir = create_run_dir(tmp_path, "model")

    assert (run_dir / "note.txt").read_text(encoding="utf-8") == run_dir.name


def test_create_run_dir_writes_custom_note(tmp_path):
    """传入备注时，实验目录应保存该备注。"""
    run_dir = create_run_dir(tmp_path, "model", note="lr sweep")

    assert (run_dir / "note.txt").read_text(encoding="utf-8") == "lr sweep"
