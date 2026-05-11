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
import copy
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

# 允许两种运行方式都能找到 `src` 包：
# 1. 推荐方式：python -m src.train
# 2. 直接方式：python src/train.py
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PVWindowDataset, fit_scalers, make_window_arrays, transform_windows
from src.evaluation_pipeline import build_model, evaluate_loaded_model, load_or_build_splits, save_artifacts
from src.features import split_feature_columns
from src.losses import elbo_loss
from src.utils import create_run_dir, describe_device, resolve_device, save_config, set_seed, setup_logger
from src.visualization import (
    plot_loss_curve,
)

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_PURPLE = "\033[35m"
ANSI_BLUE = "\033[34m"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--note", default=None, help="备注本次实验；不填时 note.txt 默认为时间戳")
    args = parser.parse_args()

    from src.utils import load_config

    config = load_config(args.config)
    if args.note is not None:
        config.setdefault("experiment", {})
        config["experiment"]["note"] = args.note
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
    run_note = config.get("experiment", {}).get("note")
    run_dir = create_run_dir(config["output_dir"], config["model"]["name"], note=run_note)
    logger = setup_logger(run_dir / "logs" / "train.log", stream_formatter=TrainingConsoleFormatter(LOG_FORMAT))
    training_started_at = time.perf_counter()
    save_config(config, run_dir / "config.yaml")
    logger.info("Device status: %s", describe_device(device))

    columns = split_feature_columns()
    # 优先使用 prepare_data 生成的固定切分；不存在时保留原始即席清洗流程。
    splits = load_or_build_splits(config)
    lookback = config["data"]["lookback"]
    horizon = config["data"]["horizon"]
    use_future_weather = config["data"].get("use_future_weather", False)

    raw_train = make_window_arrays(splits.train, columns, lookback, horizon, use_future_weather=use_future_weather)
    raw_val = make_window_arrays(splits.val, columns, lookback, horizon, use_future_weather=use_future_weather)
    # scaler 只在训练集 fit，这是防止数据泄漏的重要步骤。
    scalers = fit_scalers(raw_train)
    train_arrays = transform_windows(raw_train, scalers)
    val_arrays = transform_windows(raw_val, scalers)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))

    train_loader_kwargs = _build_loader_kwargs(config, device, shuffle=True)
    eval_loader_kwargs = _build_loader_kwargs(config, device, shuffle=False)
    if int(config["training"].get("num_workers", 0)) > 0 and train_loader_kwargs["num_workers"] == 0:
        logger.warning("Windows CUDA detected; using num_workers=0 to avoid DataLoader workers reloading CUDA DLLs.")
    train_loader = DataLoader(PVWindowDataset(train_arrays), **train_loader_kwargs)
    val_loader = DataLoader(PVWindowDataset(val_arrays), **eval_loader_kwargs)

    # 根据特征分组自动设置输入维度；当前模型使用 history/weather/direct 三路输入。
    model = build_model(config, columns, device)
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
    best_epoch = 0
    early_stop_epoch: int | None = None
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
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": config}, run_dir / "checkpoints" / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                early_stop_epoch = epoch
                logger.info("early stopping at epoch %d", epoch)
                break

    # 使用验证集上最好的权重进行 MC 推理。
    checkpoint = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # 保存表格类结果。
    plot_loss_curve(train_losses, val_losses, run_dir / "figures" / "loss_curve.png")
    save_artifacts(run_dir, columns, scalers, splits)
    evaluation_results = [evaluate_loaded_model(model, config, scalers, splits, device, "val", run_dir)]
    if not config.get("evaluation", {}).get("run_test", True):
        logger.info("validation-only run complete: %s", run_dir)
        _log_training_summary(
            logger,
            run_dir=run_dir,
            elapsed_seconds=time.perf_counter() - training_started_at,
            completed_epochs=len(train_losses),
            requested_epochs=int(config["training"]["epochs"]),
            best_epoch=best_epoch,
            best_val=best_val,
            early_stop_epoch=early_stop_epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            evaluation_results=evaluation_results,
        )
        return run_dir

    evaluation_results.append(evaluate_loaded_model(model, config, scalers, splits, device, "test", run_dir))
    logger.info("run complete: %s", run_dir)
    _log_training_summary(
        logger,
        run_dir=run_dir,
        elapsed_seconds=time.perf_counter() - training_started_at,
        completed_epochs=len(train_losses),
        requested_epochs=int(config["training"]["epochs"]),
        best_epoch=best_epoch,
        best_val=best_val,
        early_stop_epoch=early_stop_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        evaluation_results=evaluation_results,
    )
    return run_dir


class TrainingConsoleFormatter(logging.Formatter):
    """只给控制台训练日志加少量颜色，不影响 train.log 文件内容。"""

    def format(self, record: logging.LogRecord) -> str:
        console_record = copy.copy(record)
        console_record.msg = _color_training_log_message(record.getMessage())
        console_record.args = ()
        return super().format(console_record)


def _log_training_summary(logger, **summary_kwargs) -> None:
    """把训练结束摘要逐行追加到 train.log。"""
    for line in _build_training_summary_lines(**summary_kwargs):
        logger.info(line)


def _build_training_summary_lines(
    *,
    run_dir: Path,
    elapsed_seconds: float,
    completed_epochs: int,
    requested_epochs: int,
    best_epoch: int,
    best_val: float,
    early_stop_epoch: int | None,
    train_losses: list[float],
    val_losses: list[float],
    evaluation_results,
) -> list[str]:
    """生成 train.log 结尾的耗时、训练状态和指标摘要。"""
    lines = [
        "Training Summary",
        f"elapsed_time={_format_duration(elapsed_seconds)} elapsed_seconds={elapsed_seconds:.1f}",
        (
            f"completed_epochs={completed_epochs}/{requested_epochs} "
            f"best_epoch={best_epoch} best_val_loss={best_val:.6f} "
            f"early_stopping_epoch={early_stop_epoch if early_stop_epoch is not None else 'none'}"
        ),
    ]
    if train_losses:
        lines.append(f"final_train_loss={train_losses[-1]:.6f}")
    if val_losses:
        lines.append(f"final_val_loss={val_losses[-1]:.6f}")
    for result in evaluation_results:
        label = "validation" if result.split == "val" else result.split
        metrics_path = _relative_to_run_dir(result.outputs.metrics, run_dir)
        metric_text = _format_core_metrics(result.metrics)
        lines.append(f"{label}_metrics={metrics_path} {metric_text}".rstrip())
    return lines


def _format_duration(seconds: float) -> str:
    """把秒数格式化为 HH:MM:SS。"""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _relative_to_run_dir(path: Path, run_dir: Path) -> str:
    """优先显示相对 run_dir 的产物路径，日志更便于阅读。"""
    try:
        return str(Path(path).relative_to(run_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def _format_core_metrics(metrics: dict[str, float]) -> str:
    """按固定顺序输出常用评估指标，缺失时自动跳过。"""
    preferred = ["rmse", "mae", "smape", "nrmse", "crps", "picp_90", "pinaw_90", "picp_95", "pinaw_95", "nll"]
    parts = []
    for key in preferred:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.6f}")
    for key in sorted(set(metrics) - set(preferred)):
        value = metrics[key]
        if isinstance(value, int | float | np.number):
            parts.append(f"{key}={float(value):.6f}")
    return " ".join(parts)


def _color_training_log_message(message: str) -> str:
    """给训练控制台日志做克制分区着色。"""
    if message == "Training Summary":
        return _color(message, ANSI_GREEN)
    if message.startswith("Runtime options:"):
        return _highlight_key_values(message, ["batch_size", "num_workers", "pin_memory", "amp"], ANSI_BLUE)
    if message.startswith("epoch="):
        colored = _highlight_key_values(message, ["epoch"], ANSI_GREEN)
        return _highlight_key_values(colored, ["train_loss", "val_loss"], ANSI_PURPLE)
    if message.startswith("completed_epochs="):
        colored = _highlight_key_values(message, ["completed_epochs", "best_epoch", "early_stopping_epoch"], ANSI_GREEN)
        return _highlight_key_values(colored, ["best_val_loss"], ANSI_PURPLE)
    if message.startswith("final_train_loss=") or message.startswith("final_val_loss="):
        return _color(message, ANSI_PURPLE)
    if message.startswith("validation_metrics=") or message.startswith("test_metrics="):
        return _highlight_key_values(
            message,
            ["rmse", "mae", "smape", "nrmse", "crps", "picp_90", "pinaw_90", "picp_95", "pinaw_95", "nll"],
            ANSI_PURPLE,
        )
    return message


def _highlight_key_values(message: str, keys: list[str], color_code: str) -> str:
    """高亮 key=value 片段，保持其余日志默认色。"""
    key_pattern = "|".join(re.escape(key) for key in keys)
    return re.sub(rf"(?<![\w])((?:{key_pattern})=[^,\s]+)", lambda match: _color(match.group(1), color_code), message)


def _color(text: str, code: str) -> str:
    """添加 ANSI 颜色，不使用红色。"""
    return f"{code}{text}{ANSI_RESET}"


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


if __name__ == "__main__":
    main()
