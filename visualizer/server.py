"""Local server for the static visualizer and its editable JSON state.

启动方式：
1. 在项目根目录执行 `python visualizer/server.py`。
2. 浏览器打开终端输出的地址，默认是 http://127.0.0.1:5177/。
3. 如需换端口：`python visualizer/server.py --port 5178`。
4. 按 Ctrl+C 停止服务。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
HIDDEN_RUNS_FILE = ROOT / "hidden-runs.json"


def normalize_run_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/")


def normalize_run_paths(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({path for path in (normalize_run_path(value) for value in values) if path})


def read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


def resolve_run_note_path(relative_path: Any, project_root: Path = PROJECT_ROOT) -> Path:
    run_path = normalize_run_path(relative_path)
    if not run_path:
        raise ValueError("Run path is required")

    root = project_root.resolve()
    note_path = (root / run_path / "note.txt").resolve()
    if note_path != root and root not in note_path.parents:
        raise ValueError("Run note path must stay inside the project")
    if not note_path.parent.is_dir():
        suffix_matches = find_run_dirs_by_suffix(run_path, root)
        if len(suffix_matches) == 1:
            return suffix_matches[0] / "note.txt"
        if len(suffix_matches) > 1:
            raise ValueError(f"Run path is ambiguous: {run_path}")
    return note_path


def find_run_dirs_by_suffix(run_path: str, project_root: Path) -> list[Path]:
    suffix = normalize_run_path(run_path)
    if not suffix:
        return []

    run_name = suffix.split("/")[-1]
    matches: list[Path] = []
    for path in project_root.rglob(run_name):
        if not path.is_dir():
            continue
        relative = path.resolve().relative_to(project_root).as_posix()
        if relative == suffix or relative.endswith(f"/{suffix}"):
            matches.append(path.resolve())
    return sorted(matches)


def write_run_note(relative_path: Any, note: Any, project_root: Path = PROJECT_ROOT) -> Path:
    note_path = resolve_run_note_path(relative_path, project_root=project_root)
    if not note_path.parent.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {note_path.parent}")

    text = str(note if note is not None else "")
    if text and not text.endswith("\n"):
        text = f"{text}\n"
    note_path.write_text(text, encoding="utf-8")
    return note_path


class VisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        pathname = urlparse(self.path).path
        if pathname == "/api/hidden-runs":
            payload = read_json(HIDDEN_RUNS_FILE, {"hiddenRuns": []})
            self.send_json(HTTPStatus.OK, {"hiddenRuns": normalize_run_paths(payload.get("hiddenRuns"))})
            return
        super().do_GET()

    def do_PUT(self) -> None:
        pathname = urlparse(self.path).path
        if pathname == "/api/hidden-runs":
            payload = self.read_request_json()
            next_payload = {"hiddenRuns": normalize_run_paths(payload.get("hiddenRuns"))}
            write_json(HIDDEN_RUNS_FILE, next_payload)
            self.send_json(HTTPStatus.OK, next_payload)
            return
        if pathname == "/api/run-note":
            payload = self.read_request_json()
            try:
                note_path = write_run_note(payload.get("relativePath"), payload.get("note"))
            except (FileNotFoundError, ValueError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self.send_json(
                HTTPStatus.OK,
                {
                    "relativePath": normalize_run_path(payload.get("relativePath")),
                    "notePath": str(note_path.relative_to(PROJECT_ROOT).as_posix()),
                },
            )
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

    def read_request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            return {}
        return payload

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the BNN visualizer with local JSON persistence.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5177")))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    print(f"BNN visualizer running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
