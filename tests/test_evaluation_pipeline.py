"""训练后评估流水线测试。"""

from pathlib import Path

from src.evaluation_pipeline import output_names_for_split


def test_output_names_for_test_split_use_existing_filenames(tmp_path: Path):
    """测试集评估应沿用现有结果文件名，保持历史兼容。"""
    names = output_names_for_split(tmp_path, "test")

    assert names.metrics == tmp_path / "metrics" / "metrics.json"
    assert names.point_metrics == tmp_path / "metrics" / "point_metrics.csv"
    assert names.probabilistic_metrics == tmp_path / "metrics" / "probabilistic_metrics.csv"
    assert names.predictions == tmp_path / "predictions" / "test_predictions.csv"
    assert names.samples == tmp_path / "predictions" / "uncertainty_samples.npy"


def test_output_names_for_validation_split_are_prefixed(tmp_path: Path):
    """验证集独立评估不应覆盖测试集文件。"""
    names = output_names_for_split(tmp_path, "val")

    assert names.metrics == tmp_path / "metrics" / "validation_metrics.json"
    assert names.point_metrics == tmp_path / "metrics" / "validation_point_metrics.csv"
    assert names.probabilistic_metrics == tmp_path / "metrics" / "validation_probabilistic_metrics.csv"
    assert names.predictions == tmp_path / "predictions" / "validation_predictions.csv"
    assert names.samples == tmp_path / "predictions" / "validation_uncertainty_samples.npy"
