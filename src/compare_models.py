"""模型对比实验入口。

功能：
    按配置文件中的 `compare.models` 列表依次训练多个模型，并把每个模型的
    测试集指标汇总成一张 CSV 表，便于论文或报告中做模型对比实验。

使用方法：
    python -m src.compare_models --config configs/compare.yaml

输出：
    每次对比实验会创建一个独立目录，完整训练产物和汇总表都会放在其中：
        outputs/compare/YYYYMMDD-HHMMSS/
        ├── model_metrics.csv
        ├── improved_bnn/YYYYMMDD-HHMMSS/
        ├── mlp_baseline/YYYYMMDD-HHMMSS/
        ├── cnn_baseline/YYYYMMDD-HHMMSS/
        └── mc_dropout/YYYYMMDD-HHMMSS/

    汇总结果写入：
        outputs/compare/YYYYMMDD-HHMMSS/model_metrics.csv

说明：
    `configs/compare.yaml` 中的训练、数据和预测参数会被所有模型共用；
    循环时只替换 `model.name`，从而保证对比实验使用相同的数据切分和训练设置。
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.train import run_training
from src.utils import load_config
from src.visualization import plot_model_horizon_rmse, plot_model_metrics, plot_model_prediction_mean


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compare.yaml", help="模型对比实验配置文件")
    args = parser.parse_args()

    run_comparison(args.config)


def run_comparison(config_path: str | Path = "configs/compare.yaml") -> Path:
    """按 compare.yaml 中的模型列表运行对比实验。"""
    config = load_config(config_path)
    out_dir = create_compare_dir(config["output_dir"])
    rows = []
    for model_name in config["compare"]["models"]:
        # 为每个模型复制一份配置，避免循环中互相污染。
        trial_config = copy.deepcopy(config)
        trial_config["output_dir"] = str(out_dir)
        trial_config["model"]["name"] = model_name
        trial_config.setdefault("experiment", {})
        trial_config["experiment"]["note"] = f"compare {model_name}"
        run_dir = run_training(trial_config)
        metrics = pd.read_json(run_dir / "metrics" / "metrics.json", typ="series").to_dict()
        metrics["model"] = model_name
        metrics["run_dir"] = str(run_dir)
        rows.append(metrics)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(out_dir / "model_metrics.csv", index=False)
    save_comparison_figures(out_dir, metrics_frame)
    return out_dir


def create_compare_dir(output_dir: str | Path) -> Path:
    """创建本次模型对比实验的时间戳输出目录。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(output_dir) / "compare" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_comparison_figures(out_dir: str | Path, metrics_frame: pd.DataFrame) -> None:
    """在对比实验总目录下保存多模型合并图。"""
    out_path = Path(out_dir)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_model_metrics(metrics_frame, figures_dir / "compare_metrics.png")

    predictions_by_model = load_model_predictions(metrics_frame)
    if not predictions_by_model:
        return

    plot_model_prediction_mean(predictions_by_model, figures_dir / "compare_prediction_mean.png")
    horizon_rmse = build_horizon_rmse_frame(predictions_by_model)
    if not horizon_rmse.empty:
        plot_model_horizon_rmse(horizon_rmse, figures_dir / "compare_horizon_rmse.png")


def load_model_predictions(metrics_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """读取每个模型的测试集预测 CSV。"""
    predictions: dict[str, pd.DataFrame] = {}
    for row in metrics_frame.to_dict(orient="records"):
        prediction_path = Path(row["run_dir"]) / "predictions" / "test_predictions.csv"
        if prediction_path.is_file():
            predictions[str(row["model"])] = pd.read_csv(prediction_path)
    return predictions


def build_horizon_rmse_frame(predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从逐点预测结果计算多模型逐 horizon RMSE。"""
    rows = []
    for model_name, predictions in predictions_by_model.items():
        if predictions.empty:
            continue
        for horizon, group in predictions.groupby("horizon", sort=True):
            errors = group["y_mean"].astype(float) - group["y_true"].astype(float)
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "rmse": float((errors.pow(2).mean()) ** 0.5),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
