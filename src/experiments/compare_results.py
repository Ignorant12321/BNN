"""兼容旧命令的对比入口。

新入口是：
    python -m src.experiments.compare

旧入口仍可用：
    python -m src.experiments.compare_results
"""

from __future__ import annotations

from src.experiments.compare import (
    format_summary_table,
    main,
    parse_cli_run,
    parse_runs,
    read_legacy_metrics as read_metrics,
    run_compare as run_compare_results,
    run_compare_from_runs as run_compare_results_from_runs,
    write_summary,
)


if __name__ == "__main__":
    main()
