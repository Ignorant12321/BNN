"""模型对比实验入口。

当前主训练流程已经完整支持 improved_bnn。baseline 模型结构已经实现，
但统一 baseline trainer 还需要继续扩展。这个文件先保留对比实验的统一
结果表生成方式，便于后续接入 MLP、CNN 和 MC Dropout。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.train import run_training
from src.utils import load_config


def main() -> None:
    """按 compare.yaml 中的模型列表运行对比实验。"""
    config = load_config("configs/compare.yaml")
    rows = []
    for model_name in config["compare"]["models"]:
        # 为每个模型复制一份配置，避免循环中互相污染。
        trial_config = dict(config)
        trial_config["model"] = dict(config["model"])
        trial_config["model"]["name"] = model_name
        if model_name != "improved_bnn":
            # baseline 结构已经在 src.models.baselines 中，训练器接入后可移除此分支。
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
