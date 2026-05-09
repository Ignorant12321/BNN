from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.train import run_training
from src.utils import load_config


def main() -> None:
    config = load_config("configs/compare.yaml")
    rows = []
    for model_name in config["compare"]["models"]:
        trial_config = dict(config)
        trial_config["model"] = dict(config["model"])
        trial_config["model"]["name"] = model_name
        if model_name != "improved_bnn":
            print(f"{model_name} is listed for experiment tracking; implement baseline trainer before running it.")
            continue
        run_dir = run_training(trial_config)
        metrics = pd.read_json(run_dir / "metrics" / "metrics.json", typ="series").to_dict()
        metrics["model"] = model_name
        metrics["run_dir"] = str(run_dir)
        rows.append(metrics)
    out_dir = Path(config["output_dir"]) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "model_metrics.csv", index=False)


if __name__ == "__main__":
    main()
