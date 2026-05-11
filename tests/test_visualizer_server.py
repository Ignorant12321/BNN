from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_server_module():
    module_path = Path(__file__).resolve().parents[1] / "visualizer" / "server.py"
    spec = importlib.util.spec_from_file_location("visualizer_server", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_run_paths_deduplicates_and_sorts() -> None:
    server = load_server_module()

    assert server.normalize_run_paths([" outputs\\b\\20260510-193755 ", "", "outputs/b/20260510-193755"]) == [
        "outputs/b/20260510-193755"
    ]


def test_resolve_run_note_path_stays_inside_project(tmp_path: Path) -> None:
    server = load_server_module()

    note_path = server.resolve_run_note_path("outputs\\b\\20260510-193755", project_root=tmp_path)

    assert note_path == tmp_path / "outputs" / "b" / "20260510-193755" / "note.txt"


def test_resolve_run_note_path_accepts_unique_run_suffix(tmp_path: Path) -> None:
    server = load_server_module()
    run_dir = tmp_path / "outputs" / "improved_bnn" / "20260510-193755"
    run_dir.mkdir(parents=True)

    note_path = server.resolve_run_note_path("improved_bnn/20260510-193755", project_root=tmp_path)

    assert note_path == run_dir / "note.txt"


def test_resolve_run_note_path_rejects_escape(tmp_path: Path) -> None:
    server = load_server_module()

    try:
        server.resolve_run_note_path("../outside", project_root=tmp_path)
    except ValueError as error:
        assert "inside the project" in str(error)
    else:
        raise AssertionError("Expected path escape to be rejected")


def test_write_run_note_updates_note_txt(tmp_path: Path) -> None:
    server = load_server_module()
    run_dir = tmp_path / "outputs" / "b" / "20260510-193755"
    run_dir.mkdir(parents=True)

    server.write_run_note("outputs/b/20260510-193755", "saved note", project_root=tmp_path)

    assert (run_dir / "note.txt").read_text(encoding="utf-8") == "saved note\n"


def test_json_file_round_trip(tmp_path: Path) -> None:
    server = load_server_module()
    path = tmp_path / "hidden-runs.json"

    assert server.read_json(path, {"hiddenRuns": []}) == {"hiddenRuns": []}
    server.write_json(path, {"hiddenRuns": ["outputs/b/20260510-193755"]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"hiddenRuns": ["outputs/b/20260510-193755"]}
