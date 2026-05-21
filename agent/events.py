from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, run_id: str, benchmark: str, task_id: str, out_dir: str | Path):
        self.run_id = run_id
        self.benchmark = benchmark
        self.task_id = task_id
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{benchmark}__{task_id}__{run_id}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")

    def emit(self, type_: str, parent: str | None = None, **fields: Any) -> str:
        event_id = uuid.uuid4().hex[:12]
        record = {
            "event_id": event_id,
            "parent_event_id": parent,
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "type": type_,
            "ts": time.time(),
            "fields": fields,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return event_id

    def close(self) -> None:
        self._fh.close()


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def traces_dir() -> Path:
    override = os.environ.get("TRACES_DIR")
    if override:
        p = Path(override)
    else:
        here = Path(__file__).resolve().parent.parent
        p = here / "traces" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p
