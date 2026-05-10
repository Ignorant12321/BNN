"""Training evaluation metric aggregation tests."""

from __future__ import annotations

import numpy as np

from src.evaluate import evaluate_predictions


def test_evaluate_predictions_includes_crps():
    """BNN probability evaluation should report CRPS alongside NLL and intervals."""
    y_true = np.array([[0.0, 1.0]])
    mean = np.array([[0.0, 1.0]])
    std = np.array([[1.0, 1.0]])
    samples = np.array([[[0.0, 1.0]], [[0.0, 1.0]]])

    metrics = evaluate_predictions(y_true, mean, std, samples)

    assert "crps" in metrics
    assert metrics["crps"] > 0
