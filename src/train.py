"""训练主入口。

运行方式：

    python -m src.train --config configs/default.yaml

完整流程包括：

1. 读取配置并创建本次实验输出目录。
2. 读取 CSV，聚合为电站级数据，并构造基础特征。
3. 按时间顺序切分 train/val/test。
4. 构造滑动窗口并用训练集 scaler 标准化。
5. 训练 ImprovedBayesianPVNet。
6. 在测试集上进行 MC 推理，生成指标、图像、预测 CSV 和模型权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 允许两种运行方式都能找到 `src` 包：
# 1. 推荐方式：python -m src.train
# 2. 直接方式：python src/train.py
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_plant_dataframe
from src.dataset import PVWindowDataset, build_time_splits, fit_scalers, make_window_arrays, transform_windows
from src.evaluate import evaluate_predictions
from src.features import add_basic_features, split_feature_columns
from src.losses import elbo_loss
from src.metrics import horizon_metrics
from src.models.improved_bnn import ImprovedBayesianPVNet
from src.predict import interval_from_mean_std, mc_predict, select_prediction_plot_data
from src.utils import create_run_dir, describe_device, resolve_device, save_config, save_json, save_pickle, set_seed, setup_logger
from src.visualization import (
    plot_calibration_curve,
    plot_horizon_rmse,
    plot_loss_curve,
    plot_picp_pinaw,
    plot_prediction_interval,
)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    from src.utils import load_config

    config = load_config(args.config)
    run_training(config)


def run_training(config: dict) -> Path:
    """执行一次完整训练实验。

    参数:
        config: 从 YAML 读取的配置字典。

    返回:
        本次实验的输出目录路径。
    """
    import torch
    from torch.utils.data import DataLoader

    # 固定随机种子，保证同一配置下的实验尽量可复现。
    set_seed(config["seed"])
    device = resolve_device(config["training"]["device"])
    # 每次训练都会创建独立目录，避免覆盖历史实验结果。
    run_dir = create_run_dir(config["output_dir"], config["model"]["name"])
    logger = setup_logger(run_dir / "logs" / "train.log")
    save_config(config, run_dir / "config.yaml")
    logger.info("Device status: %s", describe_device(device))

    # 读取并预处理数据。这里得到的是电站级时间序列，而不是单个逆变器序列。
    df = load_plant_dataframe(
        config["data"]["generation_path"],
        config["data"]["weather_path"],
        fill_missing=config["data"].get("fill_missing", True),
    )
    df = add_basic_features(df)
    columns = split_feature_columns()
    # 先切分再构造窗口，避免训练窗口跨越到验证/测试时间段。
    splits = build_time_splits(df, config["data"]["train_ratio"], config["data"]["val_ratio"])
    lookback = config["data"]["lookback"]
    horizon = config["data"]["horizon"]
    use_future_weather = config["data"].get("use_future_weather", False)

    raw_train = make_window_arrays(splits.train, columns, lookback, horizon, use_future_weather=use_future_weather)
    raw_val = make_window_arrays(splits.val, columns, lookback, horizon, use_future_weather=use_future_weather)
    raw_test = make_window_arrays(splits.test, columns, lookback, horizon, use_future_weather=use_future_weather)
    # scaler 只在训练集 fit，这是防止数据泄漏的重要步骤。
    scalers = fit_scalers(raw_train)
    train_arrays = transform_windows(raw_train, scalers)
    val_arrays = transform_windows(raw_val, scalers)
    test_arrays = transform_windows(raw_test, scalers)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))

    train_loader_kwargs = _build_loader_kwargs(config, device, shuffle=True)
    eval_loader_kwargs = _build_loader_kwargs(config, device, shuffle=False)
    if int(config["training"].get("num_workers", 0)) > 0 and train_loader_kwargs["num_workers"] == 0:
        logger.warning("Windows CUDA detected; using num_workers=0 to avoid DataLoader workers reloading CUDA DLLs.")
    train_loader = DataLoader(PVWindowDataset(train_arrays), **train_loader_kwargs)
    val_loader = DataLoader(PVWindowDataset(val_arrays), **eval_loader_kwargs)
    test_loader = DataLoader(PVWindowDataset(test_arrays), **eval_loader_kwargs)

    # 根据特征分组自动设置输入维度；配置文件控制隐藏层规模和先验方差。
    model = ImprovedBayesianPVNet(
        history_features=len(columns.history),
        weather_features=len(columns.weather),
        time_features=len(columns.time),
        direct_features=len(columns.direct),
        horizon=horizon,
        hidden_dim=config["model"]["hidden_dim"],
        branch_dim=config["model"]["branch_dim"],
        prior_sigma=config["model"]["prior_sigma"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    amp_enabled = _amp_enabled(config, device)
    grad_scaler = _create_grad_scaler(amp_enabled)
    non_blocking = bool(config["training"].get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    logger.info(
        "Runtime options: batch_size=%s, num_workers=%s, pin_memory=%s, amp=%s",
        train_loader_kwargs["batch_size"],
        train_loader_kwargs["num_workers"],
        train_loader_kwargs["pin_memory"],
        amp_enabled,
    )

    # 记录最优验证损失，用于保存 best_model.pt 和 early stopping。
    best_val = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    patience = config["training"]["patience"]
    stale_epochs = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=config["training"]["kl_beta"],
            amp_enabled=amp_enabled,
            grad_scaler=grad_scaler,
            non_blocking=non_blocking,
        )
        val_loss = _eval_loss(
            model,
            val_loader,
            device,
            beta=config["training"]["kl_beta"],
            amp_enabled=amp_enabled,
            non_blocking=non_blocking,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info("epoch=%03d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)
        torch.save({"model_state": model.state_dict(), "config": config}, run_dir / "checkpoints" / "last_model.pt")
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": config}, run_dir / "checkpoints" / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    # 使用验证集上最好的权重进行 MC 推理。
    checkpoint = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    val_outputs = _predict_in_original_scale(model, val_loader, device, scalers, config["prediction"]["mc_samples"])
    validation_metrics = evaluate_predictions(*val_outputs)
    # 保存表格类结果。
    save_json(validation_metrics, run_dir / "metrics" / "validation_metrics.json")
    plot_loss_curve(train_losses, val_losses, run_dir / "figures" / "loss_curve.png")
    _save_artifacts(run_dir, columns, scalers, splits)
    if not config.get("evaluation", {}).get("run_test", True):
        logger.info("validation-only run complete: %s", run_dir)
        return run_dir

    target, mean, std, samples = _predict_in_original_scale(model, test_loader, device, scalers, config["prediction"]["mc_samples"])
    metrics = evaluate_predictions(target, mean, std, samples)
    save_json(metrics, run_dir / "metrics" / "metrics.json")
    pd.DataFrame(horizon_metrics(target, mean)).to_csv(run_dir / "metrics" / "point_metrics.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics" / "probabilistic_metrics.csv", index=False)
    _save_predictions(run_dir, raw_test.target_times, target, mean, std, samples)
    # 保存论文中常用的可视化图。
    lower90, upper90 = interval_from_mean_std(mean, std, 0.90)
    lower95, upper95 = interval_from_mean_std(mean, std, 0.95)
    plot_config = config.get("prediction", {}).get("plot", {})
    view90 = select_prediction_plot_data(
        raw_test.target_times,
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
        raw_test.target_times,
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
    logger.info("prediction interval plot selection: %s", view90["reason"])
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
    logger.info("run complete: %s", run_dir)
    return run_dir


def _build_loader_kwargs(config: dict, device, shuffle: bool, platform: str | None = None) -> dict:
    """根据配置生成 DataLoader 参数。"""
    training = config.get("training", {})
    num_workers = int(training.get("num_workers", 0))
    platform_name = sys.platform if platform is None else platform
    if (
        platform_name.startswith("win")
        and device.type == "cuda"
        and num_workers > 0
        and not bool(training.get("allow_windows_cuda_workers", False))
    ):
        num_workers = 0
    pin_memory = bool(training.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    kwargs = {
        "batch_size": int(training["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
    return kwargs


def _amp_enabled(config: dict, device) -> bool:
    """只在 CUDA 设备上启用自动混合精度。"""
    return bool(config.get("training", {}).get("amp", False)) and device.type == "cuda"


def _create_grad_scaler(enabled: bool):
    """兼容不同 PyTorch 版本创建 GradScaler。"""
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _train_epoch(model, loader, optimizer, device, beta: float, amp_enabled: bool = False, grad_scaler=None, non_blocking: bool = False) -> float:
    """训练一个 epoch，并返回平均 loss。"""
    import torch

    model.train()
    losses = []
    for batch in loader:
        # DataLoader 返回的每个 batch 是一个字典，所有张量都搬到同一设备。
        batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
        target = batch.pop("target")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            mean, log_var = model(batch)
            # ELBO = Gaussian NLL + beta * KL。KL 来自所有贝叶斯层。
            loss = elbo_loss(mean, log_var, target, model.kl_loss(), beta=beta, num_batches=len(loader))
        if amp_enabled and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _eval_loss(model, loader, device, beta: float, amp_enabled: bool = False, non_blocking: bool = False) -> float:
    """在验证集上计算平均 loss，不更新参数。"""
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
            target = batch.pop("target")
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                mean, log_var = model(batch)
                loss = elbo_loss(mean, log_var, target, model.kl_loss(), beta=beta, num_batches=len(loader))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    """把标准化后的目标值恢复到原始 AC_POWER 尺度。"""
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def _predict_in_original_scale(model, loader, device, scalers, mc_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """执行 MC 推理并把目标、均值、标准差恢复到真实功率尺度。"""
    pred = mc_predict(model, loader, device, mc_samples=mc_samples)
    target = _inverse_target(pred["target"], scalers["target"])
    mean = _inverse_target(pred["mean"], scalers["target"])
    std = pred["std"] * float(scalers["target"].scale_[0])
    samples = np.stack([_inverse_target(s, scalers["target"]) for s in pred["samples"]], axis=0)
    return target, mean, std, samples


def _save_predictions(run_dir: Path, target_times: np.ndarray, target: np.ndarray, mean: np.ndarray, std: np.ndarray, samples: np.ndarray) -> None:
    """保存测试集逐样本、逐 horizon 的预测结果。"""
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
    pd.DataFrame(rows).to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)
    np.save(run_dir / "predictions" / "uncertainty_samples.npy", samples)


def _save_artifacts(run_dir: Path, columns, scalers, splits) -> None:
    """保存复现实验所需的工件。"""
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


if __name__ == "__main__":
    main()
