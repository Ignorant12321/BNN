"""轻量图表输出。"""

from __future__ import annotations

from pathlib import Path
from math import isfinite


def write_metrics_bar_svg(rows: list[dict[str, str]], path: Path, metric: str = "test_rmse") -> None:
    """不依赖额外库，写出简单 SVG 柱状图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for row in rows:
        try:
            value = float(row.get(metric, row.get(metric.removeprefix("test_"), "nan")))
        except ValueError:
            continue
        if isfinite(value):
            values.append((row["label"], value))
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"480\" height=\"80\"></svg>\n", encoding="utf-8")
        return
    max_value = max(value for _, value in values) or 1.0
    width = 560
    row_height = 32
    height = 40 + row_height * len(values)
    lines = [f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">"]
    lines.append("<style>text{font-family:Arial;font-size:12px}.bar{fill:#4f8cc9}</style>")
    lines.append(f"<text x=\"12\" y=\"20\">{metric}</text>")
    for index, (label, value) in enumerate(values):
        y = 36 + index * row_height
        bar_width = int(360 * value / max_value)
        lines.append(f"<text x=\"12\" y=\"{y + 14}\">{label}</text>")
        lines.append(f"<rect class=\"bar\" x=\"120\" y=\"{y}\" width=\"{bar_width}\" height=\"18\"/>")
        lines.append(f"<text x=\"{128 + bar_width}\" y=\"{y + 14}\">{value:.4f}</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
