"""模型对比实验入口。

功能：
    按配置文件中的 `compare.models` 列表依次训练多个模型，并把每个模型的
    测试集指标汇总成一张 CSV 表，便于论文或报告中做模型对比实验。

使用方法：
    python -m src.compare_models --config configs/compare.yaml

输出：
    每个模型的完整训练产物仍会写入各自的 run 目录，例如：
        outputs/improved_bnn/YYYYMMDD-HHMMSS/
        outputs/mlp_baseline/YYYYMMDD-HHMMSS/

    汇总结果写入：
        outputs/comparison/model_metrics.csv

说明：
    `configs/compare.yaml` 中的训练、数据和预测参数会被所有模型共用；
    循环时只替换 `model.name`，从而保证对比实验使用相同的数据切分和训练设置。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from src.train import run_training
from src.utils import load_config


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compare.yaml", help="模型对比实验配置文件")
    args = parser.parse_args()

    run_comparison(args.config)


def run_comparison(config_path: str | Path = "configs/compare.yaml") -> Path:
    """按 compare.yaml 中的模型列表运行对比实验。"""
    config = load_config(config_path)
    rows = []
    for model_name in config["compare"]["models"]:
        # 为每个模型复制一份配置，避免循环中互相污染。
        trial_config = copy.deepcopy(config)
        trial_config["model"]["name"] = model_name
        trial_config.setdefault("experiment", {})
        trial_config["experiment"]["note"] = f"compare {model_name}"
        run_dir = run_training(trial_config)
        metrics = pd.read_json(run_dir / "metrics" / "metrics.json", typ="series").to_dict()
        metrics["model"] = model_name
        metrics["run_dir"] = str(run_dir)
        rows.append(metrics)
    out_dir = Path(config["output_dir"]) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "model_metrics.csv", index=False)
    return out_dir


if __name__ == "__main__":
    main()
