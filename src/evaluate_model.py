"""独立模型评估入口。

运行方式：

    python -m src.evaluate_model --run-dir outputs/improved_bnn/YYYYMMDD-HHMMSS

该命令读取指定实验目录中的 config、best_model.pt 和 scaler，在验证集或
测试集上重新执行 MC 推理并导出指标、预测 CSV 和图像。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation_pipeline import evaluate_run_dir


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="训练输出目录，例如 outputs/improved_bnn/20260510-120000")
    parser.add_argument("--split", default="test", choices=["val", "validation", "test", "both"], help="要评估的数据集")
    parser.add_argument("--checkpoint", default="best_model.pt", help="checkpoints/ 下的模型文件名")
    parser.add_argument("--mc-samples", type=int, default=None, help="覆盖 config.yaml 中的 MC 采样次数")
    args = parser.parse_args()

    results = evaluate_run_dir(args.run_dir, split=args.split, checkpoint_name=args.checkpoint, mc_samples=args.mc_samples)
    for result in results:
        print(f"{result.split} metrics written to: {result.outputs.metrics}")


if __name__ == "__main__":
    main()
