"""训练后评估与预测结果导出。

本模块把验证/测试集评估从训练入口中拆出来，供 `train.py` 和
`evaluate_model.py` 共用。训练仍然可以在结束后自动评估；也可以后续
对同一个 run_dir 单独重新评估，例如调整 MC 采样次数或重新生成图表。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_plant_dataframe
from src.dataset import PVWindowDataset, TimeSplits, build_time_splits, make_window_arrays, transform_windows
from src.evaluate import evaluate_predictions
from src.features import FeatureColumns, add_basic_features, split_feature_columns
from src.metrics import horizon_metrics
from src.predict import interval_from_mean_std, mc_predict, select_prediction_plot_data
from src.prepare_data import load_processed_splits
from src.utils import load_config, load_pickle, resolve_device, save_json
from src.visualization import plot_calibration_curve, plot_horizon_rmse, plot_picp_pinaw, plot_prediction_interval


@dataclass(frozen=True)
class EvaluationOutputNames:
    """某个数据 split 的评估输出文件路径。"""

    metrics: Path
    point_metrics: Path
    probabilistic_metrics: Path
    predictions: Path
    samples: Path


@dataclass
class EvaluationResult:
    """单次 split 评估的主要结果。"""

    split: str
    metrics: dict[str, float]
    outputs: EvaluationOutputNames


def output_names_for_split(run_dir: str | Path, split: str) -> EvaluationOutputNames:
    """返回验证集或测试集评估产物的文件名。"""
    run_path = Path(run_dir)
    normalized = normalize_split(split)
    if normalized == "test":
        return EvaluationOutputNames(
            metrics=run_path / "metrics" / "metrics.json",
            point_metrics=run_path / "metrics" / "point_metrics.csv",
            probabilistic_metrics=run_path / "metrics" / "probabilistic_metrics.csv",
            predictions=run_path / "predictions" / "test_predictions.csv",
            samples=run_path / "predictions" / "uncertainty_samples.npy",
        )
    return EvaluationOutputNames(
        metrics=run_path / "metrics" / "validation_metrics.json",
        point_metrics=run_path / "metrics" / "validation_point_metrics.csv",
        probabilistic_metrics=run_path / "metrics" / "validation_probabilistic_metrics.csv",
        predictions=run_path / "predictions" / "validation_predictions.csv",
        samples=run_path / "predictions" / "validation_uncertainty_samples.npy",
    )


def normalize_split(split: str) -> str:
    """规范化 split 名称。"""
    value = split.lower()
    if value in {"validation", "valid"}:
        return "val"
    if value not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    return value


def load_or_build_splits(config: dict) -> TimeSplits:
    """优先读取 prepare_data 产物，否则从原始 CSV 即席构造切分。"""
    splits = load_processed_splits(config)
    if splits is not None:
        return splits
    df = load_plant_dataframe(
        config["data"]["generation_path"],
        config["data"]["weather_path"],
        fill_missing=config["data"].get("fill_missing", True),
    )
    df = add_basic_features(df)
    return build_time_splits(df, config["data"]["train_ratio"], config["data"]["val_ratio"])


def build_model(config: dict, columns: FeatureColumns, device):
    """按配置构造模型。当前独立评估仅支持 improved_bnn。"""
    from src.models.improved_bnn import ImprovedBayesianPVNet

    if config["model"].get("name", "improved_bnn") != "improved_bnn":
        raise ValueError("evaluate_model currently supports only improved_bnn checkpoints")
    return ImprovedBayesianPVNet(
        history_features=len(columns.history),
        weather_features=len(columns.weather),
        direct_features=len(columns.direct),
        horizon=config["data"]["horizon"],
        hidden_dim=config["model"]["hidden_dim"],
        branch_dim=config["model"]["branch_dim"],
        prior_sigma=config["model"]["prior_sigma"],
    ).to(device)


def predict_in_original_scale(model, loader, device, scalers, mc_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """执行 MC 推理并把目标、均值、标准差恢复到真实 AC_POWER 尺度。"""
    pred = mc_predict(model, loader, device, mc_samples=mc_samples)
    target = inverse_target(pred["target"], scalers["target"])
    mean = inverse_target(pred["mean"], scalers["target"])
    std = pred["std"] * float(scalers["target"].scale_[0])
    samples = np.stack([inverse_target(s, scalers["target"]) for s in pred["samples"]], axis=0)
    return target, mean, std, samples


def resolve_mc_samples(config: dict, default: int = 50) -> int:
    """Return configured MC sample count, falling back to prediction default."""
    return int(config.get("prediction", {}).get("mc_samples", default))


def inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    """把标准化后的目标值恢复到原始 AC_POWER 尺度。"""
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def save_artifacts(run_dir: Path, columns, scalers, splits) -> None:
    """保存复现实验所需的特征、scaler 和切分信息。"""
    from src.utils import save_pickle

    save_pickle(scalers["history"], run_dir / "artifacts" / "scaler_x.pkl")
    save_pickle(scalers["target"], run_dir / "artifacts" / "scaler_y.pkl")
    save_pickle(scalers, run_dir / "artifacts" / "all_scalers.pkl")
    save_json(columns.__dict__, run_dir / "artifacts" / "feature_columns.json")
    split_info = {
        "train_start": str(splits.train["DATE_TIME"].min()),
        "train_end": str(splits.train["DATE_TIME"].max()),
        "val_start": str(splits.val["DATE_TIME"].min()),
        "val_end": str(splits.val["DATE_TIME"].max()),
        "test_start": str(splits.test["DATE_TIME"].min()),
        "test_end": str(splits.test["DATE_TIME"].max()),
    }
    save_json(split_info, run_dir / "artifacts" / "split_info.json")


def evaluate_loaded_model(
    model,
    config: dict,
    scalers,
    splits: TimeSplits,
    device,
    split: str,
    run_dir: str | Path,
) -> EvaluationResult:
    """评估已加载到内存的模型，并保存指标、预测和图像。"""
    import torch
    from torch.utils.data import DataLoader

    normalized = normalize_split(split)
    run_path = Path(run_dir)
    columns = split_feature_columns()
    source_df = splits.test if normalized == "test" else splits.val
    raw_windows = make_window_arrays(
        source_df,
        columns,
        config["data"]["lookback"],
        config["data"]["horizon"],
        use_future_weather=config["data"].get("use_future_weather", False),
    )
    arrays = transform_windows(raw_windows, scalers)
    loader = DataLoader(
        PVWindowDataset(arrays),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        target, mean, std, samples = predict_in_original_scale(
            model,
            loader,
            device,
            scalers,
            resolve_mc_samples(config),
        )
    metrics = evaluate_predictions(target, mean, std, samples)
    outputs = output_names_for_split(run_path, normalized)
    save_evaluation_outputs(run_path, normalized, raw_windows.target_times, target, mean, std, samples, metrics, outputs, config)
    return EvaluationResult(split=normalized, metrics=metrics, outputs=outputs)


def evaluate_run_dir(
    run_dir: str | Path,
    split: str = "test",
    checkpoint_name: str = "best_model.pt",
    mc_samples: int | None = None,
) -> list[EvaluationResult]:
    """从输出目录加载 checkpoint，并独立重新评估指定 split。"""
    import torch

    run_path = Path(run_dir)
    config = load_config(run_path / "config.yaml")
    if mc_samples is not None:
        config.setdefault("prediction", {})
        config["prediction"]["mc_samples"] = mc_samples
    device = resolve_device(config["training"].get("device", "cpu"))
    columns = split_feature_columns()
    model = build_model(config, columns, device)
    checkpoint = torch.load(run_path / "checkpoints" / checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    scalers = load_pickle(run_path / "artifacts" / "all_scalers.pkl")
    splits = load_or_build_splits(config)

    requested = ["val", "test"] if split == "both" else [normalize_split(split)]
    return [
        evaluate_loaded_model(model, config, scalers, splits, device, one_split, run_path)
        for one_split in requested
    ]


def save_evaluation_outputs(
    run_dir: Path,
    split: str,
    target_times: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    samples: np.ndarray,
    metrics: dict[str, float],
    outputs: EvaluationOutputNames,
    config: dict,
) -> None:
    """保存单个 split 的指标、预测 CSV、采样数组和测试集图像。"""
    save_json(metrics, outputs.metrics)
    pd.DataFrame(horizon_metrics(target, mean)).to_csv(outputs.point_metrics, index=False)
    pd.DataFrame([metrics]).to_csv(outputs.probabilistic_metrics, index=False)
    save_predictions(outputs.predictions, target_times, target, mean, std)
    np.save(outputs.samples, samples)
    if split == "test":
        save_test_figures(run_dir, target_times, target, mean, std, config)


def save_predictions(path: Path, target_times: np.ndarray, target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    """保存逐样本、逐 horizon 的预测结果。"""
    rows = []
    for i in range(target.shape[0]):
        for h in range(target.shape[1]):
            rows.append(
                {
                    "sample": i,
                    "horizon": h + 1,
                    "target_time": str(target_times[i, h]),
                    "y_true": target[i, h],
                    "y_mean": mean[i, h],
                    "y_std": std[i, h],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_test_figures(run_dir: Path, target_times: np.ndarray, target: np.ndarray, mean: np.ndarray, std: np.ndarray, config: dict) -> None:
    """保存测试集论文常用图像。"""
    lower90, upper90 = interval_from_mean_std(mean, std, 0.90)
    lower95, upper95 = interval_from_mean_std(mean, std, 0.95)
    plot_config = config.get("prediction", {}).get("plot", {})
    view90 = select_prediction_plot_data(
        target_times,
        target,
        mean,
        lower90,
        upper90,
        start_time=plot_config.get("start_time"),
        end_time=plot_config.get("end_time"),
        prefer_daylight=plot_config.get("prefer_daylight", True),
        daylight_threshold=plot_config.get("daylight_threshold", 1.0),
        max_points=plot_config.get("max_points", 160),
    )
    view95 = select_prediction_plot_data(
        target_times,
        target,
        mean,
        lower95,
        upper95,
        start_time=plot_config.get("start_time"),
        end_time=plot_config.get("end_time"),
        prefer_daylight=plot_config.get("prefer_daylight", True),
        daylight_threshold=plot_config.get("daylight_threshold", 1.0),
        max_points=plot_config.get("max_points", 160),
    )
    plot_prediction_interval(
        view90["y_true"],
        view90["mean"],
        view90["lower"],
        view90["upper"],
        run_dir / "figures" / "prediction_interval_90.png",
        times=view90["times"],
    )
    plot_prediction_interval(
        view95["y_true"],
        view95["mean"],
        view95["lower"],
        view95["upper"],
        run_dir / "figures" / "prediction_interval_95.png",
        times=view95["times"],
    )
    plot_horizon_rmse(target, mean, run_dir / "figures" / "horizon_rmse.png")
    plot_picp_pinaw(target, {"90%": (lower90, upper90), "95%": (lower95, upper95)}, run_dir / "figures" / "picp_pinaw.png")
    plot_calibration_curve(target, mean, std, run_dir / "figures" / "calibration_curve.png")
