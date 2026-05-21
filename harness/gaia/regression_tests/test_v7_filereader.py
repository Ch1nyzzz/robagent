"""Smoke tests for v7's file reader (no LLM calls)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.events import EventLog, new_run_id  # noqa: E402
from agent.v7.tools.file_reader import read_gaia_file  # noqa: E402


def _make_log(tmp_path):
    log = EventLog(run_id=new_run_id(), benchmark="test", task_id="t1", out_dir=str(tmp_path))
    return log, log.emit("run.started")


def test_read_text_file(tmp_path):
    # write a small txt file in a temp HF-cache-like layout
    fake_path = tmp_path / "snapshot" / "2023" / "validation" / "x.txt"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text("hello world\nthis is a test")
    # patch resolver
    import agent.v7.tools.file_reader as fr

    orig = fr.gaia_file_path
    fr.gaia_file_path = lambda tid, fn: str(fake_path) if fn == "x.txt" else None
    try:
        log, root = _make_log(tmp_path)
        out = read_gaia_file("t1", "x.txt", log=log, parent=root)
        log.close()
    finally:
        fr.gaia_file_path = orig
    assert out["ok"] is True
    assert "hello world" in out["content"]
    assert out["source_id"].startswith("S_file_")


def test_read_json_file(tmp_path):
    fake_path = tmp_path / "x.json"
    fake_path.write_text(json.dumps({"k": "v", "arr": [1, 2, 3]}))
    import agent.v7.tools.file_reader as fr

    orig = fr.gaia_file_path
    fr.gaia_file_path = lambda tid, fn: str(fake_path) if fn == "x.json" else None
    try:
        log, root = _make_log(tmp_path)
        out = read_gaia_file("t1", "x.json", log=log, parent=root)
        log.close()
    finally:
        fr.gaia_file_path = orig
    assert out["ok"] is True
    assert '"k": "v"' in out["content"]


def test_read_unsupported_extension(tmp_path):
    fake_path = tmp_path / "x.mp3"
    fake_path.write_bytes(b"audio")
    import agent.v7.tools.file_reader as fr

    orig = fr.gaia_file_path
    fr.gaia_file_path = lambda tid, fn: str(fake_path) if fn == "x.mp3" else None
    try:
        log, root = _make_log(tmp_path)
        out = read_gaia_file("t1", "x.mp3", log=log, parent=root)
        log.close()
    finally:
        fr.gaia_file_path = orig
    assert out["ok"] is False
    assert "unsupported_extension" in out["reason"]


def test_read_missing_path(tmp_path):
    import agent.v7.tools.file_reader as fr

    orig = fr.gaia_file_path
    fr.gaia_file_path = lambda tid, fn: None
    try:
        log, root = _make_log(tmp_path)
        out = read_gaia_file("t1", "missing.txt", log=log, parent=root)
        log.close()
    finally:
        fr.gaia_file_path = orig
    assert out["ok"] is False
    assert out["reason"] == "path_not_found"
