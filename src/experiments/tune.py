"""Optuna 超参数调参入口。

功能：
    基于一个模型训练配置生成多个 trial，按验证集指标选择最佳配置。
    默认使用 SQLite storage，因此重复运行同一个调参配置会从已有 study 继续。

使用：
    python -m src.experiments.tune --config configs/tune/bnn.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import yaml

from src.artifacts.run_io import write_run_note
from src.config import load_config
from src.experiments.train import run_training


PARAM_ALIASES = {
    "lr": "training.lr",
    "weight_decay": "training.weight_decay",
    "kl_beta": "training.kl_beta",
    "epochs": "training.epochs",
    "batch_size": "training.batch_size",
    "hidden_dim": "model.hidden_dim",
    "branch_dim": "model.branch_dim",
    "conv_kernel": "model.conv_kernel",
    "dropout": "model.dropout",
    "n_samples": "evaluation.n_samples",
}


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Tune PV forecasting model hyperparameters with Optuna.")
    parser.add_argument("--config", default="configs/tune/bnn.yaml", help="调参配置文件")
    parser.add_argument("--note", default=None, help="写入调参输出目录 note.txt 的备注；默认写入输出目录名")
    args = parser.parse_args()
    result = run_tuning(args.config, note=args.note)
    print(f"Tuning output: {result['tuning_dir']}")
    print(f"Completed trials: {result['completed_trials']}/{result['target_trials']}")
    if result.get("best_value") is not None:
        print(f"Best {result['metric']}: {result['best_value']}")
        print(f"Best run: {result.get('best_run', '')}")


def run_tuning(config_path: str | Path, note: str | None = None) -> dict[str, Any]:
    """读取调参配置并执行 Optuna study。"""
    tune_config_path = Path(config_path)
    tune_config = load_tune_config(tune_config_path)
    name = str(tune_config.get("name", "tuning"))
    output_root = Path(tune_config.get("output_dir", "outputs"))
    tuning_dir = output_root / "tuning" / name
    tuning_dir.mkdir(parents=True, exist_ok=True)
    write_run_note(tuning_dir, note)
    save_yaml(tune_config, tuning_dir / "tuning_config.yaml")

    optuna = import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna.trial import TrialState

    storage = str(tune_config.get("storage") or sqlite_storage_url(tuning_dir / "study.db"))
    study = optuna.create_study(
        study_name=str(tune_config.get("study_name", name)),
        storage=storage,
        direction=str(tune_config.get("direction", "minimize")),
        load_if_exists=True,
    )
    target_trials = int(tune_config.get("n_trials", 20))
    completed_trials = count_trials(study, TrialState.COMPLETE)
    remaining_trials = max(0, target_trials - completed_trials)
    if remaining_trials:
        study.optimize(
            make_objective(tune_config, tune_config_path, tuning_dir),
            n_trials=remaining_trials,
            callbacks=[lambda study, trial: print_trial_summary(study, trial, str(tune_config.get("metric", "val_rmse")))],
        )

    write_trials_csv(study, tuning_dir / "trials.csv")
    result = {
        "tuning_dir": tuning_dir,
        "target_trials": target_trials,
        "completed_trials": count_trials(study, TrialState.COMPLETE),
        "metric": str(tune_config.get("metric", "val_rmse")),
        "best_value": None,
        "best_run": "",
    }
    if count_trials(study, TrialState.COMPLETE):
        best_trial = study.best_trial
        best_config = build_best_config(tune_config, tune_config_path, best_trial.params)
        save_yaml(best_config, tuning_dir / "best_config.yaml")
        best_run = str(best_trial.user_attrs.get("run_dir", ""))
        (tuning_dir / "best_run.txt").write_text(best_run + "\n", encoding="utf-8")
        result["best_value"] = best_trial.value
        result["best_run"] = best_run
    return result


def make_objective(tune_config: dict[str, Any], tune_config_path: Path, tuning_dir: Path):
    """构造 Optuna objective。"""
    base_config_path = resolve_base_config_path(tune_config, tune_config_path)
    base_config = load_config(base_config_path)
    search_space = tune_config.get("search_space", {})
    metric_name = str(tune_config.get("metric", "val_rmse"))

    def objective(trial) -> float:
        trial_config, params = build_trial_config(base_config, search_space, trial)
        trial_config.setdefault("tuning", {})
        trial_config["tuning"].update(
            {
                "study_name": str(tune_config.get("study_name", tune_config.get("name", "tuning"))),
                "trial_number": int(trial.number),
                "params": params,
            }
        )
        run_dir = tuning_dir / "runs" / f"trial-{trial.number:04d}"
        trial_config["run_dir"] = str(run_dir)
        run_dir = run_trial_with_config(tune_config, trial_config, run_dir, int(trial.number))
        value = read_metric_from_run(run_dir, metric_name)
        trial.set_user_attr("run_dir", str(run_dir))
        return value

    return objective


def run_trial_with_config(tune_config: dict[str, Any], trial_config: dict[str, Any], run_dir: Path, trial_number: int) -> Path:
    """Run one Optuna trial with the configured training runner."""
    runner = str(tune_config.get("runner", "training"))
    if runner in {"training", "default"}:
        return run_training(trial_config)
    if runner == "recursive_bnn_4h":
        result = run_recursive_bnn_4h_trial(
            trial_config,
            run_dir=run_dir,
            note=f"Recursive 4h BNN Optuna trial {trial_number}",
        )
        return Path(result.run_dir)
    raise ValueError(f"unsupported tuning runner: {runner}")


def run_recursive_bnn_4h_trial(config: dict[str, Any], run_dir: Path, note: str | None = None):
    """Run one recursive 4h BNN tuning trial."""
    from src.experiments.train_bnn_recursive_4h import run_recursive_training

    return run_recursive_training(config, run_dir=run_dir, note=note)


def print_trial_summary(study, trial, metric_name: str) -> None:
    """打印比 Optuna 默认日志更适合阅读的 trial 摘要。"""
    print(flush=True)
    print(f"Tuning Trial {trial.number}", flush=True)
    print("-" * 48, flush=True)
    print(f"State      : {trial.state.name}", flush=True)
    if trial.value is not None:
        print(f"{metric_name:<10}: {trial.value:.6f}", flush=True)
    print(f"Run        : {trial.user_attrs.get('run_dir', '')}", flush=True)
    if trial.params:
        print("Parameters :", flush=True)
        for name, value in sorted(trial.params.items()):
            print(f"  {name}: {value}", flush=True)
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None and best_trial.value is not None:
        print(f"Best Trial : {best_trial.number}", flush=True)
        print(f"Best Value : {best_trial.value:.6f}", flush=True)


def build_trial_config(base_config: dict[str, Any], search_space: dict[str, Any], trial) -> tuple[dict[str, Any], dict[str, Any]]:
    """从 search_space 采样并写入训练配置。"""
    trial_config = copy.deepcopy(base_config)
    params: dict[str, Any] = {}
    for raw_name, spec in search_space.items():
        value = suggest_value(trial, raw_name, spec)
        set_config_value(trial_config, normalize_param_path(raw_name), value)
        params[raw_name] = value
    return trial_config, params


def suggest_value(trial, name: str, spec: dict[str, Any]):
    """根据单个搜索空间定义向 Optuna trial 取样。"""
    if not isinstance(spec, dict):
        raise ValueError(f"search_space.{name} must be a mapping")
    kind = str(spec.get("type", "categorical"))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if kind == "log_float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "log_int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=True)
    raise ValueError(f"unsupported search_space.{name}.type: {kind}")


def normalize_param_path(name: str) -> str:
    """把短参数名映射成训练配置中的点路径。"""
    return PARAM_ALIASES.get(name, name)


def set_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """按点路径写入嵌套配置。"""
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"config path {dotted_path!r} crosses a non-mapping value")
    target[parts[-1]] = value


def read_metric_from_run(run_dir: str | Path, metric_name: str) -> float:
    """从训练产物的 metrics.csv 读取形如 val_rmse 的指标。"""
    if "_" not in metric_name:
        raise ValueError("metric must use split_metric form, for example val_rmse")
    metrics_path = Path(run_dir) / "metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if f"{row.get('split')}_{row.get('metric')}" == metric_name:
                return float(row["value"])
    raise ValueError(f"metric {metric_name!r} not found in {metrics_path}")


def build_best_config(tune_config: dict[str, Any], tune_config_path: Path, params: dict[str, Any]) -> dict[str, Any]:
    """根据最佳 trial 参数生成最终训练配置快照。"""
    best_config = load_config(resolve_base_config_path(tune_config, tune_config_path))
    for raw_name, value in params.items():
        set_config_value(best_config, normalize_param_path(raw_name), value)
    return best_config


def write_trials_csv(study, path: Path) -> None:
    """写出 Optuna trials 摘要。"""
    param_names = sorted({name for trial in study.trials for name in trial.params})
    fields = ["number", "state", "value", "run_dir", *param_names]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": "" if trial.value is None else trial.value,
                "run_dir": trial.user_attrs.get("run_dir", ""),
            }
            row.update({name: trial.params.get(name, "") for name in param_names})
            writer.writerow(row)


def resolve_base_config_path(tune_config: dict[str, Any], tune_config_path: Path) -> Path:
    """解析 base_config，支持相对于调参配置文件的路径。"""
    base_config = Path(str(tune_config["base_config"]))
    if not base_config.is_absolute():
        base_config = tune_config_path.parent / base_config
    return base_config


def load_tune_config(path: Path) -> dict[str, Any]:
    """读取调参 YAML。"""
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("top-level tuning YAML object must be a mapping")
    return payload


def save_yaml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sqlite_storage_url(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def count_trials(study, state) -> int:
    return sum(1 for trial in study.trials if trial.state == state)


def import_optuna():
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Optuna is required for tuning. Install it with `pip install optuna`.") from error
    return optuna


if __name__ == "__main__":
    main()
