"""Local server for browsing BNN training outputs.

Run from the project root:

    python visualizer/server.py

Then open the printed local URL. By default the server scans ``outputs`` and
serves files from the project directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
project_root_text = str(PROJECT_ROOT.resolve())
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

from src.evaluation.metrics import normalization_scale

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")
FIGURE_GROUP_ORDER = {"loss": 0, "predict": 1}


def normalize_path(path: str | Path) -> str:
    return Path(path).as_posix().strip("/")


def is_run_dir(path: Path) -> bool:
    return (path / "config.yaml").is_file() or (path / "metrics.csv").is_file() or (path / "epoch_history.csv").is_file()


def is_comparison_dir(path: Path) -> bool:
    return (path / "model_metrics.csv").is_file()


def is_timestamp_comparison_dir(path: Path) -> bool:
    return TIMESTAMP_RE.match(path.name) is not None and path.parent.name == "comparisons" and path.parent.parent.name == "outputs"


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_project_root_on_syspath(project_root: Path = PROJECT_ROOT) -> None:
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_note(path: Path) -> str:
    note_path = path / "note.txt"
    if not note_path.is_file():
        return ""
    return note_path.read_text(encoding="utf-8").strip()


def read_metrics(path: Path) -> dict[str, float]:
    metrics_path = path / "metrics.csv"
    if not metrics_path.is_file():
        return {}
    metrics: dict[str, float] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            split = str(row.get("split", "")).strip()
            metric = str(row.get("metric", "")).strip()
            value = row.get("value")
            if not split or not metric or value in (None, ""):
                continue
            try:
                metrics[f"{split}_{metric}"] = float(value)
            except ValueError:
                continue
    return metrics


def read_epoch_history(path: Path) -> list[dict[str, float]]:
    history_path = path / "epoch_history.csv"
    if not history_path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with history_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                rows.append({key: float(value) for key, value in row.items() if value not in (None, "")})
            except ValueError:
                continue
    return rows


def collect_figures(run_dir: Path, project_root: Path) -> list[dict[str, str]]:
    figures_dir = run_dir / "figures"
    if not figures_dir.is_dir():
        return []
    figures = []
    candidates = []
    for path in (item for item in figures_dir.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
        group = figure_group(path.name)
        if group is None:
            continue
        candidates.append((FIGURE_GROUP_ORDER[group], path.name, group, path))
    for _, _, group, path in sorted(candidates):
        relative = safe_relative(path, project_root)
        figures.append(
            {
                "name": path.name,
                "group": group,
                "path": relative,
                "url": f"/files/{quote(relative)}",
            }
        )
    return figures


def figure_group(name: str) -> str | None:
    lower_name = name.lower()
    if lower_name.startswith("loss"):
        return "loss"
    if lower_name.startswith("prediction"):
        return "predict"
    return None


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "run"


def prediction_summary(comparison_dir: Path, label: str) -> dict[str, float | int] | None:
    predictions_dir = comparison_dir / "predictions"
    candidates = [predictions_dir / f"{label}.csv", predictions_dir / f"{safe_filename(label)}.csv"]
    prediction_path = next((path for path in candidates if path.is_file()), None)
    if prediction_path is None:
        return None
    rows = 0
    horizons: set[str] = set()
    target_sum = 0.0
    prediction_sum = 0.0
    absolute_error_sum = 0.0
    targets: list[float] = []
    interval90_hits = 0
    interval95_hits = 0
    interval90_width_sum = 0.0
    interval95_width_sum = 0.0
    interval_rows = 0
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                target = float(row["target"])
                prediction = float(row["mean"])
            except (KeyError, TypeError, ValueError):
                continue
            rows += 1
            horizons.add(str(row.get("horizon", "")))
            target_sum += target
            prediction_sum += prediction
            absolute_error_sum += abs(target - prediction)
            targets.append(target)
            try:
                lower90 = float(row["lower_90"])
                upper90 = float(row["upper_90"])
                lower95 = float(row["lower_95"])
                upper95 = float(row["upper_95"])
            except (KeyError, TypeError, ValueError):
                continue
            interval_rows += 1
            interval90_hits += int(lower90 <= target <= upper90)
            interval95_hits += int(lower95 <= target <= upper95)
            interval90_width_sum += upper90 - lower90
            interval95_width_sum += upper95 - lower95
    if rows == 0:
        return None
    summary = {
        "rows": rows,
        "horizons": len({value for value in horizons if value != ""}),
        "targetMean": target_sum / rows,
        "predictionMean": prediction_sum / rows,
        "mae": absolute_error_sum / rows,
    }
    if interval_rows:
        scale = normalization_scale(np.array(targets, dtype=float))
        summary.update(
            {
                "picp90": interval90_hits / interval_rows,
                "pinaw90": (interval90_width_sum / interval_rows) / scale,
                "picp95": interval95_hits / interval_rows,
                "pinaw95": (interval95_width_sum / interval_rows) / scale,
            }
        )
    return summary


def collect_run(run_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = read_yaml(run_dir / "config.yaml")
    model = config.get("model", {}).get("name", run_dir.parent.name) if isinstance(config.get("model"), dict) else run_dir.parent.name
    return {
        "label": run_dir.name,
        "model": str(model),
        "path": safe_relative(run_dir, project_root),
        "note": read_note(run_dir),
        "config": config,
        "metrics": read_metrics(run_dir),
        "history": read_epoch_history(run_dir),
        "figures": collect_figures(run_dir, project_root),
    }


def collect_comparison(comparison_dir: Path, project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    metrics_path = comparison_dir / "model_metrics.csv"
    if not metrics_path.is_file():
        return []
    runs: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            label = str(row.get("label") or comparison_dir.name).strip()
            run_dir_text = str(row.get("run_dir") or "").strip()
            source_run_dir = (project_root / run_dir_text).resolve() if run_dir_text else None
            config = read_yaml(source_run_dir / "config.yaml") if source_run_dir and source_run_dir.is_dir() else {}
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key in {"label", "model", "run_dir"} or value in (None, ""):
                    continue
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            runs.append(
                {
                    "label": label,
                    "model": str(row.get("model") or config.get("model", {}).get("name", "model")),
                    "path": f"{safe_relative(comparison_dir, project_root)}#{label}",
                    "note": f"comparison {comparison_dir.name}",
                    "config": config,
                    "metrics": metrics,
                    "predictionSummary": prediction_summary(comparison_dir, label),
                    "history": [],
                    "figures": [],
                }
            )
    return runs


def count_comparison_rows(comparison_dir: Path) -> int:
    metrics_path = comparison_dir / "model_metrics.csv"
    if not metrics_path.is_file():
        return 0
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def comparison_summary(comparison_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        "name": comparison_dir.name,
        "path": safe_relative(comparison_dir, project_root),
        "note": read_note(comparison_dir),
        "runCount": count_comparison_rows(comparison_dir),
    }


def list_comparisons(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    comparisons_root = project_root / "outputs" / "comparisons"
    if not comparisons_root.is_dir():
        return []
    comparisons = [
        path
        for path in comparisons_root.iterdir()
        if path.is_dir() and is_timestamp_comparison_dir(path) and (path / "model_metrics.csv").is_file()
    ]
    return [comparison_summary(path, project_root) for path in sorted(comparisons, key=lambda item: item.name, reverse=True)]


def resolve_comparison_dir(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    clean = normalize_path(value)
    if not clean:
        raise ValueError("comparison path is required")
    if "/" not in clean:
        clean = f"outputs/comparisons/{clean}"
    path = (project_root / clean).resolve()
    try:
        relative = path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("comparison path must stay inside the project") from error
    if len(relative.parts) != 3 or relative.parts[0] != "outputs" or relative.parts[1] != "comparisons":
        raise ValueError("comparison path must be outputs/comparisons/<timestamp>")
    if TIMESTAMP_RE.match(relative.parts[2]) is None:
        raise ValueError("comparison path must be outputs/comparisons/<timestamp>")
    return path


def read_comparison(value: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    comparison_dir = resolve_comparison_dir(value, project_root)
    if not comparison_dir.is_dir() or not (comparison_dir / "model_metrics.csv").is_file():
        raise FileNotFoundError(f"comparison not found: {safe_relative(comparison_dir, project_root)}")
    return {
        "comparison": comparison_summary(comparison_dir, project_root),
        "runs": collect_comparison(comparison_dir, project_root),
        "figures": collect_figures(comparison_dir, project_root),
    }


def list_train_runs(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    train_root = project_root / "outputs" / "train"
    if not train_root.is_dir():
        return []
    run_dirs = sorted({path for path in train_root.rglob("*") if path.is_dir() and is_run_dir(path)}, key=lambda item: item.as_posix())
    return [collect_run(path, project_root) for path in run_dirs]


def create_comparison(
    runs: list[dict[str, str]],
    project_root: Path = PROJECT_ROOT,
    name: str = "visualizer",
    split: str = "test",
    note: str | None = None,
    compare_runner=None,
) -> dict[str, Any]:
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")
    parsed_runs: list[dict[str, str]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        path = str(run.get("path") or "").strip()
        label = str(run.get("label") or "").strip()
        if not path:
            raise ValueError(f"runs[{index}].path must be a non-empty string")
        parsed = {"path": path}
        if label:
            parsed["label"] = label
        parsed_runs.append(parsed)
    runner_runs = parsed_runs
    if compare_runner is None:
        ensure_project_root_on_syspath(project_root)
        from src.experiments.compare import run_compare_from_runs

        compare_runner = run_compare_from_runs
        runner_runs = [_resolve_runner_run_path(run, project_root) for run in parsed_runs]
    out_dir = compare_runner(runner_runs, name=name or "visualizer", output_dir=project_root / "outputs", split=split or "test", note=note)
    return read_comparison(safe_relative(Path(out_dir), project_root), project_root)


def _resolve_runner_run_path(run: dict[str, str], project_root: Path) -> dict[str, str]:
    path = Path(run["path"])
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("run path must stay inside the project") from error
    result = dict(run)
    result["path"] = str(resolved)
    return result


def discover_runs(output_root: str | Path = PROJECT_ROOT / "outputs", project_root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(output_root)
    base = project_root or root.parent
    if not root.is_dir():
        return []
    run_dirs = sorted({path for path in root.rglob("*") if path.is_dir() and is_run_dir(path)}, key=lambda item: item.as_posix())
    comparison_dirs = sorted({path for path in root.rglob("*") if path.is_dir() and is_comparison_dir(path)}, key=lambda item: item.as_posix())
    runs: list[dict[str, Any]] = []
    for comparison_dir in comparison_dirs:
        runs.extend(collect_comparison(comparison_dir, base))
    runs.extend(collect_run(path, base) for path in run_dirs)
    return runs


def discover_default_runs(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    return discover_runs(project_root / "outputs", project_root)


class VisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, project_root: Path = PROJECT_ROOT, **kwargs: Any) -> None:
        self.project_root = project_root.resolve()
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        pathname = parsed.path
        if pathname == "/api/runs":
            self.send_json({"runs": discover_default_runs(self.project_root)})
            return
        if pathname == "/api/comparisons":
            self.send_json({"comparisons": list_comparisons(self.project_root)})
            return
        if pathname == "/api/comparison":
            query = parse_qs(parsed.query)
            comparison_path = query.get("path", [""])[0]
            try:
                self.send_json(read_comparison(comparison_path, self.project_root))
            except (FileNotFoundError, ValueError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return
        if pathname == "/api/train-runs":
            self.send_json({"runs": list_train_runs(self.project_root)})
            return
        if pathname == "/api/output-dirs":
            self.send_json({"outputDirs": self.list_output_dirs()})
            return
        if pathname.startswith("/files/"):
            self.serve_project_file(pathname.removeprefix("/files/"))
            return
        super().do_GET()

    def do_POST(self) -> None:
        pathname = urlparse(self.path).path
        if pathname == "/api/comparisons":
            payload = self.read_request_json()
            try:
                result = create_comparison(
                    payload.get("runs", []),
                    self.project_root,
                    name=str(payload.get("name") or "visualizer"),
                    split=str(payload.get("split") or "test"),
                    note=payload.get("note"),
                )
            except (FileNotFoundError, ValueError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(result, status=HTTPStatus.CREATED)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def translate_path(self, path: str) -> str:
        pathname = unquote(urlparse(path).path)
        if pathname == "/":
            pathname = "/index.html"
        resolved = (ROOT / pathname.lstrip("/")).resolve()
        if resolved != ROOT and ROOT not in resolved.parents:
            return str(ROOT / "__forbidden__")
        return str(resolved)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        if path.endswith(".js"):
            return "text/javascript; charset=utf-8"
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type and mime_type.startswith(("text/", "application/json")):
            return f"{mime_type}; charset=utf-8"
        return mime_type or "application/octet-stream"

    def list_output_dirs(self) -> list[str]:
        outputs = self.project_root / "outputs"
        if not outputs.is_dir():
            return []
        return [safe_relative(path, self.project_root) for path in sorted(outputs.iterdir()) if path.is_dir()]

    def serve_project_file(self, encoded_relative: str) -> None:
        relative = unquote(encoded_relative)
        try:
            path = (self.project_root / relative).resolve()
            path.relative_to(self.project_root)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as file:
            self.wfile.write(file.read())

    def read_request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the BNN visualizer.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5177")))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    def handler(*handler_args: Any, **handler_kwargs: Any) -> VisualizerHandler:
        return VisualizerHandler(*handler_args, project_root=PROJECT_ROOT, **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"BNN visualizer running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
