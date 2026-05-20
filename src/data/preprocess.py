"""数据预处理命令。

功能：
    读取原始发电 CSV 和天气 CSV，完成逆变器聚合、天气合并，并输出
    `processed_dir/plant_frame.csv`。

使用：
    python -m src.data.preprocess --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data.pv import load_plant_dataframe


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="预处理光伏原始 CSV 数据。")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径")
    args = parser.parse_args()
    output_path = run_preprocess(load_config(args.config))
    print(f"预处理数据已保存到: {output_path}")


def run_preprocess(config: dict) -> Path:
    """执行数据预处理并返回输出 CSV 路径。"""
    data = config["data"]
    processed_dir = Path(data.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "plant_frame.csv"
    frame = load_plant_dataframe(data["generation_path"], data["weather_path"])
    frame.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    main()
