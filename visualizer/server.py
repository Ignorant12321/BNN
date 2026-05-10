"""Local server for the static visualizer and its editable JSON state."""

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
HIDDEN_RUNS_FILE = ROOT / "hidden-runs.json"
RUN_NOTES_FILE = ROOT / "run-notes.json"


def normalize_run_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/")


def normalize_run_paths(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({path for path in (normalize_run_path(value) for value in values) if path})


def normalize_run_notes(notes: Any) -> dict[str, str]:
    if not isinstance(notes, dict):
        return {}
    normalized = {
        normalize_run_path(path): str(note if note is not None else "")
        for path, note in notes.items()
        if normalize_run_path(path)
    }
    return dict(sorted(normalized.items()))


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


class VisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        pathname = urlparse(self.path).path
        if pathname == "/api/hidden-runs":
            payload = read_json(HIDDEN_RUNS_FILE, {"hiddenRuns": []})
            self.send_json(HTTPStatus.OK, {"hiddenRuns": normalize_run_paths(payload.get("hiddenRuns"))})
            return
        if pathname == "/api/run-notes":
            payload = read_json(RUN_NOTES_FILE, {"notes": {}})
            self.send_json(HTTPStatus.OK, {"notes": normalize_run_notes(payload.get("notes"))})
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
        if pathname == "/api/run-notes":
            payload = self.read_request_json()
            next_payload = {"notes": normalize_run_notes(payload.get("notes"))}
            write_json(RUN_NOTES_FILE, next_payload)
            self.send_json(HTTPStatus.OK, next_payload)
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
