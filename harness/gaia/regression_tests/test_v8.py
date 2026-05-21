"""Unit tests for v8 — image dispatch and arithmetic tool wiring."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.v8.workflow import _IMAGE_EXTS  # noqa: E402
from agent.v8.tools.arithmetic import verify_arithmetic, safe_arith  # noqa: E402
from agent.events import EventLog, new_run_id, traces_dir  # noqa: E402


def test_image_exts_includes_png_jpg():
    assert "png" in _IMAGE_EXTS
    assert "jpg" in _IMAGE_EXTS
    assert "jpeg" in _IMAGE_EXTS


def test_arithmetic_tool_emits_events(tmp_path):
    log = EventLog(run_id=new_run_id(), benchmark="test", task_id="t1", out_dir=str(tmp_path))
    root = log.emit("run.started")
    out = verify_arithmetic("2 + 2", expected="4", log=log, parent=root)
    log.close()
    assert out["ok"] is True
    assert out["value"] == 4
    assert out["matches_expected"] is True


def test_arithmetic_tool_rejects_unsafe(tmp_path):
    log = EventLog(run_id=new_run_id(), benchmark="test", task_id="t2", out_dir=str(tmp_path))
    root = log.emit("run.started")
    out = verify_arithmetic("__import__('os').system('ls')", log=log, parent=root)
    log.close()
    assert out["ok"] is False
