"""Optuna 自动调参入口。

运行方式：

    python -m src.tune

每个 trial 会复制 `configs/tuning.yaml`，替换若干超参数，然后调用完整训练流程。
目标函数默认最小化测试集 RMSE。由于每个 trial 都会真实训练模型，运行时间会
明显长于单次训练。
"""

from __future__ import annotations

import copy

from src.train import run_training
from src.utils import load_config


def main() -> None:
    """执行 Optuna 超参数搜索。"""
    import optuna

    base = load_config("configs/tuning.yaml")

    def objective(trial):
        """单个 trial 的目标函数。"""
        config = copy.deepcopy(base)
        # 模型容量相关参数。
        config["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        config["model"]["branch_dim"] = trial.suggest_categorical("branch_dim", [32, 64, 128])
        # 优化器和贝叶斯 KL 权重相关参数。
        config["training"]["lr"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        config["training"]["kl_beta"] = trial.suggest_float("kl_beta", 1e-5, 1e-2, log=True)
        run_dir = run_training(config)
        import json

        with open(run_dir / "metrics" / "metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)["rmse"]

    # direction=minimize 表示 RMSE 越小越好。
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=base["tuning"]["n_trials"])
    print(study.best_params)


if __name__ == "__main__":
    main()
