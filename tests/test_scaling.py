import numpy as np

from src.data.pv import WindowArrays
from src.data.scaling import fit_window_scaler, inverse_target_prediction, transform_window_arrays_by_split


def test_window_scaler_fits_train_split_only_and_restores_target_predictions():
    train = WindowArrays(
        history=np.array([[[0.0], [10.0]], [[20.0], [30.0]]], dtype=np.float32),
        weather=np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        direct=np.array([[0.0], [30.0]], dtype=np.float32),
        target=np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
    )
    val = WindowArrays(
        history=np.array([[[1000.0], [1010.0]]], dtype=np.float32),
        weather=np.array([[[100.0], [200.0]]], dtype=np.float32),
        direct=np.array([[1000.0]], dtype=np.float32),
        target=np.array([[1000.0, 1100.0]], dtype=np.float32),
    )

    scaler = fit_window_scaler({"train": train, "val": val})
    transformed = transform_window_arrays_by_split({"train": train, "val": val}, scaler)

    assert scaler["target"]["mean"] == 250.0
    assert scaler["target"]["std"] == np.std(train.target)
    np.testing.assert_allclose(np.mean(transformed["train"].target), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(transformed["train"].target), 1.0, atol=1e-6)
    np.testing.assert_allclose(transformed["val"].target, (val.target - 250.0) / np.std(train.target))

    restored_mean, restored_log_var = inverse_target_prediction(
        transformed["val"].target,
        np.log(np.ones_like(transformed["val"].target, dtype=np.float32)),
        scaler,
    )

    np.testing.assert_allclose(restored_mean, val.target)
    np.testing.assert_allclose(np.exp(restored_log_var), np.ones_like(val.target) * np.std(train.target) ** 2, rtol=1e-5)
