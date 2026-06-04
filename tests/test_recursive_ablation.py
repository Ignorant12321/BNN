import numpy as np

from src.data.pv import WindowArrays
from src.experiments.ablate_bnn_recursive_4h import (
    AblationSpec,
    ablation_config,
    zero_feature_group_arrays,
)


def test_pv_usibnn_recursive_ablation_keeps_features_and_marks_zero_group():
    config = {
        "model": {"name": "pv_usibnn_recursive"},
        "data": {
            "features": {
                "history": ["AC_POWER"],
                "weather": ["IRRADIATION"],
                "direct": ["AC_POWER"],
                "target": "AC_POWER",
            }
        },
    }

    result = ablation_config(config, AblationSpec("no_history", "history"))

    assert result["data"]["features"]["history"] == ["AC_POWER"]
    assert result["strategy"]["zero_feature_group"] == "history"


def test_zero_feature_group_arrays_only_zeros_requested_input():
    arrays = WindowArrays(
        history=np.ones((2, 3, 1), dtype=np.float32),
        weather=np.ones((2, 4, 2), dtype=np.float32) * 2.0,
        direct=np.ones((2, 1), dtype=np.float32) * 3.0,
        target=np.ones((2, 4), dtype=np.float32) * 4.0,
        target_time=np.array([["t1", "t2", "t3", "t4"], ["t5", "t6", "t7", "t8"]]),
    )

    result = zero_feature_group_arrays(arrays, "weather")

    assert np.all(result.history == 1.0)
    assert np.all(result.weather == 0.0)
    assert np.all(result.direct == 3.0)
    assert np.all(result.target == 4.0)
    assert result.target_time is arrays.target_time
