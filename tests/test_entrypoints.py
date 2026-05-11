"""命令行入口回归测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_script_help_works_when_executed_directly() -> None:
    """直接运行 src/train.py 时也应能解析包内导入。"""
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(repo_root / "src" / "train.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_evaluate_model_help_works_when_executed_directly() -> None:
    """独立评估入口应能显示命令行帮助。"""
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(repo_root / "src" / "evaluate_model.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout
    assert "--split" in result.stdout


def test_compare_models_help_works_without_starting_training() -> None:
    """模型对比入口应支持 --help，避免误触后直接开始训练。"""
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-m", "src.compare_models", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
