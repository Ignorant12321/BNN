from __future__ import annotations

import copy

from src.train import run_training
from src.utils import load_config


def main() -> None:
    import optuna

    base = load_config("configs/tuning.yaml")

    def objective(trial):
        config = copy.deepcopy(base)
        config["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        config["model"]["branch_dim"] = trial.suggest_categorical("branch_dim", [32, 64, 128])
        config["training"]["lr"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        config["training"]["kl_beta"] = trial.suggest_float("kl_beta", 1e-5, 1e-2, log=True)
        run_dir = run_training(config)
        import json

        with open(run_dir / "metrics" / "metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=base["tuning"]["n_trials"])
    print(study.best_params)


if __name__ == "__main__":
    main()
