from src.environment import describe_torch_environment


def test_describe_torch_environment_returns_status_mapping():
    status = describe_torch_environment()

    assert "available" in status
    assert "cuda_available" in status
    assert "device" in status
