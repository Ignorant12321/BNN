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

    train_power = np.concatenate(
        [
            train.history.reshape(-1),
            train.direct.reshape(-1),
            train.target.reshape(-1),
        ]
    )
    expected_power_mean = float(np.mean(train_power))
    expected_power_std = float(np.std(train_power))
    assert scaler["target"]["mean"] == expected_power_mean
    assert scaler["target"]["std"] == expected_power_std
    assert scaler["history"]["mean"] == [expected_power_mean]
    assert scaler["history"]["std"] == [expected_power_std]
    assert scaler["direct"]["mean"] == [expected_power_mean]
    assert scaler["direct"]["std"] == [expected_power_std]
    transformed_train_power = np.concatenate(
        [
            transformed["train"].history.reshape(-1),
            transformed["train"].direct.reshape(-1),
            transformed["train"].target.reshape(-1),
        ]
    )
    np.testing.assert_allclose(np.mean(transformed_train_power), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(transformed_train_power), 1.0, atol=1e-6)
    np.testing.assert_allclose(transformed["val"].target, (val.target - expected_power_mean) / expected_power_std)
    np.testing.assert_allclose(transformed["val"].history, (val.history - expected_power_mean) / expected_power_std)
    np.testing.assert_allclose(transformed["val"].direct, (val.direct - expected_power_mean) / expected_power_std)

    restored_mean, restored_log_var = inverse_target_prediction(
        transformed["val"].target,
        np.log(np.ones_like(transformed["val"].target, dtype=np.float32)),
        scaler,
    )

    np.testing.assert_allclose(restored_mean, val.target)
    np.testing.assert_allclose(np.exp(restored_log_var), np.ones_like(val.target) * expected_power_std**2, rtol=1e-5)


def test_window_scaler_handles_empty_history_features():
    train = WindowArrays(
        history=np.zeros((2, 0, 0), dtype=np.float32),
        weather=np.ones((2, 1, 1), dtype=np.float32),
        direct=np.ones((2, 1), dtype=np.float32),
        target=np.ones((2, 1), dtype=np.float32),
    )

    scaler = fit_window_scaler({"train": train})

    assert scaler["history"] == {"mean": [], "std": []}
    transformed = transform_window_arrays_by_split({"train": train}, scaler)["train"]
    assert transformed.history.shape == (2, 0, 0)
