import importlib.util

import pytest


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


torch_required = pytest.mark.skipif(
    not has_torch(),
    reason="torch is not installed in this environment",
)
