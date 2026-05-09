"""通用工具函数测试。"""

from src.utils import describe_device


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
