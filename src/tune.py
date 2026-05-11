"""Optuna 自动调参入口。

运行方式：

    python -m src.tune

每个 trial 会复制 `configs/tuning.yaml`，替换若干超参数，然后调用完整训练流程。
目标函数默认最小化验证集 RMSE。由于每个 trial 都会真实训练模型，运行时间会
明显长于单次训练。
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.train import run_training
from src.utils import load_config, save_config, save_json


METRIC_ORDER = ["rmse", "mae", "smape", "nrmse", "crps", "picp_90", "pinaw_90", "picp_95", "pinaw_95", "nll"]
ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_PURPLE = "\033[35m"
ANSI_BLUE = "\033[34m"


def main() -> None:
    """执行 Optuna 超参数搜索。"""
    import optuna

    configure_optuna_logging(optuna)
    base = load_config("configs/tuning.yaml")
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    objective_metric = resolve_objective_metric(base)
    tuning_started_at = time.perf_counter()

    def objective(trial):
        """单个 trial 的目标函数。"""
        trial_started_at = time.perf_counter()
        config = prepare_trial_config(base, tuning_run_name=run_name)
        apply_trial_suggestions(config, trial)
        print(format_trial_config(trial.number, config, color=True), flush=True)
        run_dir = run_training(config)
        metrics = load_validation_metrics(run_dir)
        print(
            format_trial_result(
                trial.number,
                run_dir,
                metrics,
                objective_metric,
                elapsed_seconds=time.perf_counter() - trial_started_at,
                color=True,
            ),
            flush=True,
        )
        return float(metrics[objective_metric])

    # direction=minimize 表示目标指标越小越好。
    study = create_tuning_study(optuna, base)
    n_trials = count_remaining_trials(study, target_n_trials=base["tuning"]["n_trials"])
    if n_trials > 0:
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[lambda study, trial: print(format_optuna_trial_log(study, trial, color=True), flush=True)],
        )
    else:
        print(f"Study already has {len(study.trials)} trials; skipping optimization.")
    export_dir = export_study_results(study, base, run_name=run_name)
    print(
        format_study_summary(
            study,
            export_dir,
            objective_metric=objective_metric,
            elapsed_seconds=time.perf_counter() - tuning_started_at,
            color=True,
        )
    )


def prepare_trial_config(base_config: dict, platform: str = sys.platform, tuning_run_name: str | None = None) -> dict:
    """复制基础配置，并让调参 trial 只评估验证集。"""
    config = copy.deepcopy(base_config)
    if tuning_run_name is not None:
        config["output_dir"] = str(Path(base_config["output_dir"]) / "tuning" / tuning_run_name)
    config.setdefault("evaluation", {})
    config["evaluation"]["run_test"] = False
    if platform == "win32":
        config.setdefault("training", {})
        config["training"]["num_workers"] = 0
        config["training"]["persistent_workers"] = False
    return config


def create_tuning_study(optuna_module, base_config: dict):
    """创建可持久化的 Optuna study，以支持中断后继续调参。"""
    tuning_config = base_config.get("tuning", {})
    storage = tuning_config.get("storage")
    if storage:
        ensure_sqlite_storage_parent(storage)

    kwargs = {
        "direction": tuning_config.get("direction", "minimize"),
    }
    if tuning_config.get("study_name") is not None:
        kwargs["study_name"] = tuning_config["study_name"]
    if storage:
        kwargs["storage"] = storage
        kwargs["load_if_exists"] = tuning_config.get("load_if_exists", True)

    return optuna_module.create_study(**kwargs)


def configure_optuna_logging(optuna_module) -> None:
    """关闭 Optuna 默认 INFO 单行日志，避免挤在格式化摘要中间。"""
    optuna_module.logging.set_verbosity(optuna_module.logging.WARNING)
    optuna_module.logging.disable_default_handler()


def ensure_sqlite_storage_parent(storage: str) -> None:
    """为本地 SQLite storage 创建父目录。"""
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return

    db_path = storage[len(prefix) :]
    if db_path == ":memory:":
        return

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def count_remaining_trials(study, target_n_trials: int) -> int:
    """计算续跑时还需要补充的 trial 数量。"""
    return max(0, target_n_trials - len(study.trials))


def resolve_objective_metric(config: dict) -> str:
    """读取 Optuna 目标指标；默认保持历史 RMSE 行为。"""
    return str(config.get("tuning", {}).get("objective_metric", "rmse"))


def apply_trial_suggestions(config: dict, trial) -> None:
    """根据 tuning.search_space 采样超参数并写回 trial 配置。"""
    search_space = config.get("tuning", {}).get("search_space", {})
    hidden_dim_choices = search_space.get("hidden_dim", [64, 128, 256])
    branch_dim_choices = search_space.get("branch_dim", [32, 64, 128])
    lr_space = search_space.get("lr", {"low": 1e-4, "high": 3e-3, "log": True})
    kl_beta_space = search_space.get("kl_beta", {"low": 1e-5, "high": 1e-2, "log": True})

    config.setdefault("model", {})
    config.setdefault("training", {})
    # 模型容量相关参数。
    config["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", hidden_dim_choices)
    config["model"]["branch_dim"] = trial.suggest_categorical("branch_dim", branch_dim_choices)
    # 优化器和贝叶斯 KL 权重相关参数。
    config["training"]["lr"] = trial.suggest_float(
        "lr",
        lr_space["low"],
        lr_space["high"],
        log=lr_space.get("log", True),
    )
    config["training"]["kl_beta"] = trial.suggest_float(
        "kl_beta",
        kl_beta_space["low"],
        kl_beta_space["high"],
        log=kl_beta_space.get("log", True),
    )


def format_trial_config(trial_number: int, config: dict, color: bool = False) -> str:
    """生成 trial 开始前的参数摘要，避免控制台输出挤成一行。"""
    model = config.get("model", {})
    training = config.get("training", {})
    trial_label = _color(f"Optuna Trial {trial_number + 1}", ANSI_GREEN, color)
    lines = [
        "",
        f"========== {trial_label} ==========",
        _color("Sampled params:", ANSI_BLUE, color),
        _color(f"  hidden_dim: {_format_value(model.get('hidden_dim'))}", ANSI_BLUE, color),
        _color(f"  branch_dim: {_format_value(model.get('branch_dim'))}", ANSI_BLUE, color),
        _color(f"  lr: {_format_value(training.get('lr'))}", ANSI_BLUE, color),
        _color(f"  kl_beta: {_format_value(training.get('kl_beta'))}", ANSI_BLUE, color),
        "Training options:",
        f"  epochs: {_format_value(training.get('epochs'))}",
        f"  patience: {_format_value(training.get('patience'))}",
        "=====================================",
    ]
    return "\n".join(lines)


def format_trial_result(
    trial_number: int,
    run_dir: str | Path,
    metrics: dict[str, float],
    objective_metric: str,
    elapsed_seconds: float,
    color: bool = False,
) -> str:
    """生成单个 trial 结束后的摘要。"""
    trial_label = _color(f"Optuna Trial {trial_number + 1}", ANSI_GREEN, color)
    lines = [
        "",
        f"========== {trial_label} Complete ==========",
        f"elapsed_time: {format_duration(elapsed_seconds)}",
        _color(f"objective_metric: {objective_metric}", ANSI_PURPLE, color),
        _color(f"objective_value: {_format_metric(metrics[objective_metric])}", ANSI_PURPLE, color),
        _color("Validation metrics:", ANSI_PURPLE, color),
    ]
    lines.extend(format_metric_lines(metrics, color=color))
    lines.extend(
        [
            f"run_dir: {run_dir}",
            "==============================================",
        ]
    )
    return "\n".join(lines)


def format_study_summary(
    study,
    export_dir: str | Path,
    objective_metric: str = "rmse",
    elapsed_seconds: float | None = None,
    color: bool = False,
) -> str:
    """生成调参结束摘要，多行展示 best trial 和 best params。"""
    lines = [
        "",
        "Tuning Summary",
    ]
    if elapsed_seconds is not None:
        lines.append(f"elapsed_time: {format_duration(elapsed_seconds)}")
    lines.extend(
        [
            _color(f"objective_metric: {objective_metric}", ANSI_PURPLE, color),
            "",
            "Best trial:",
            _color(f"  number: {study.best_trial.number + 1}", ANSI_GREEN, color),
            _color(f"  value: {_format_value(study.best_value)}", ANSI_PURPLE, color),
            "",
            _color("Best params:", ANSI_BLUE, color),
        ]
    )
    for name, value in study.best_params.items():
        lines.append(_color(f"  {name}: {_format_value(value)}", ANSI_BLUE, color))
    lines.extend(
        [
            "",
            "Tuning results exported to:",
            f"  {export_dir}",
        ]
    )
    return "\n".join(lines)


def format_optuna_trial_log(study, trial, color: bool = False) -> str:
    """生成替代 Optuna 默认 INFO 的 trial 完成摘要。"""
    trial_label = _color(f"Optuna Trial {trial.number + 1}", ANSI_GREEN, color)
    lines = [
        "",
        f"========== {trial_label} Finished ==========",
        _color(f"value: {_format_value(trial.value)}", ANSI_PURPLE, color),
        "Best trial:",
        _color(f"  number: {study.best_trial.number + 1}", ANSI_GREEN, color),
        _color(f"  value: {_format_value(study.best_value)}", ANSI_PURPLE, color),
        _color("Params:", ANSI_BLUE, color),
    ]
    for name, value in trial.params.items():
        lines.append(_color(f"  {name}: {_format_value(value)}", ANSI_BLUE, color))
    lines.append("============================================")
    return "\n".join(lines)


def format_metric_lines(metrics: dict[str, float], color: bool = False) -> list[str]:
    """按固定顺序格式化验证集指标。"""
    lines = []
    for key in METRIC_ORDER:
        if key in metrics:
            lines.append(_color(f"  {key}: {_format_metric(metrics[key])}", ANSI_PURPLE, color))
    for key in sorted(set(metrics) - set(METRIC_ORDER)):
        value = metrics[key]
        if isinstance(value, int | float):
            lines.append(_color(f"  {key}: {_format_metric(value)}", ANSI_PURPLE, color))
    return lines


def format_duration(seconds: float) -> str:
    """把秒数格式化为 HH:MM:SS。"""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_value(value) -> str:
    """统一格式化控制台里的标量值。"""
    return str(value)


def _format_metric(value: float) -> str:
    """统一格式化控制台里的指标值。"""
    return f"{float(value):.6f}"


def _color(text: str, code: str, enabled: bool) -> str:
    """按需添加 ANSI 颜色，默认保持纯文本，便于测试和日志复用。"""
    if not enabled:
        return text
    return f"{code}{text}{ANSI_RESET}"


def merge_best_params(base_config: dict, best_params: dict) -> dict:
    """把 Optuna 最佳参数合并到一份训练配置副本中。"""
    config = copy.deepcopy(base_config)
    config.setdefault("model", {})
    config.setdefault("training", {})
    if "hidden_dim" in best_params:
        config["model"]["hidden_dim"] = best_params["hidden_dim"]
    if "branch_dim" in best_params:
        config["model"]["branch_dim"] = best_params["branch_dim"]
    if "lr" in best_params:
        config["training"]["lr"] = best_params["lr"]
    if "kl_beta" in best_params:
        config["training"]["kl_beta"] = best_params["kl_beta"]
    return config


def export_study_results(study, base_config: dict, run_name: str | None = None) -> Path:
    """导出 Optuna 调参结果和可直接复用的最佳配置。"""
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dir = Path(base_config["output_dir"]) / "tuning" / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "objective_metric": resolve_objective_metric(base_config),
        "best_trial": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }
    save_json(summary, export_dir / "best_params.json")
    study.trials_dataframe().to_csv(export_dir / "trials.csv", index=False)
    save_config(merge_best_params(base_config, study.best_params), export_dir / "best_config.yaml")
    return export_dir


def load_objective_metric(run_dir: str | Path, metric: str = "rmse") -> float:
    """读取 Optuna 目标指标。

    调参必须使用验证集指标，避免测试集参与模型选择。
    """
    return float(load_validation_metrics(run_dir)[metric])


def load_validation_metrics(run_dir: str | Path) -> dict[str, float]:
    """读取完整验证集指标。"""
    path = Path(run_dir) / "metrics" / "validation_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {key: float(value) for key, value in metrics.items()}


if __name__ == "__main__":
    main()
