from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Iterator


def _tau2_data_root() -> Path:
    root = os.environ.get("TAU2_DATA_ROOT")
    if root:
        p = Path(root)
        if p.exists():
            return p
    try:
        import tau2  # type: ignore

        pkg_root = Path(tau2.__file__).resolve().parent
        for candidate in [
            pkg_root.parent.parent / "data" / "tau2",
            pkg_root.parent / "data" / "tau2",
            pkg_root / "data",
        ]:
            if candidate.exists():
                return candidate
    except Exception:
        pass
    here = Path(__file__).resolve().parent.parent.parent
    for candidate in [
        here / "tau2-bench-src" / "data" / "tau2",
        here / "tau2-bench" / "data" / "tau2",
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Cannot locate tau2-bench data. Set TAU2_DATA_ROOT, install tau2 package, "
        "or clone sierra-research/tau2-bench into ./tau2-bench-src."
    )


def list_domains() -> list[str]:
    root = _tau2_data_root()
    domains_dir = root / "domains"
    if not domains_dir.exists():
        return []
    return sorted([p.name for p in domains_dir.iterdir() if p.is_dir()])


def iter_tasks(
    domain: str | None = None,
    domain_limits: dict[str, int] | None = None,
    seed: int = 0,
) -> Iterator[dict]:
    """Iterate tau2-bench tasks across all domains (or a single domain).

    domain_limits: optional cap per domain; sampled deterministically by seed.
    Yields dicts with keys: task_id, domain, raw (full task dict).
    """
    import random

    root = _tau2_data_root()
    domains = [domain] if domain else list_domains()
    for d in domains:
        tasks_path = root / "domains" / d / "tasks.json"
        if not tasks_path.exists():
            for alt in (root / "domains" / d).glob("tasks*.json"):
                tasks_path = alt
                break
        if not tasks_path.exists():
            continue
        data = json.loads(tasks_path.read_text(encoding="utf-8"))
        tasks = data if isinstance(data, list) else data.get("tasks", [])
        cap = (domain_limits or {}).get(d)
        if cap is not None and len(tasks) > cap:
            rng = random.Random(f"{d}:{seed}")
            tasks = rng.sample(tasks, cap)
        for t in tasks:
            tid = t.get("id") or t.get("task_id") or t.get("name") or "unknown"
            yield {"task_id": f"{d}::{tid}", "domain": d, "raw": t}


def build_prompt(task: dict) -> str:
    raw = task["raw"]
    us = raw.get("user_scenario") or {}
    ins = us.get("instructions") if isinstance(us, dict) else None
    parts: list[str] = []
    if isinstance(ins, dict):
        for k in ("reason_for_call", "task_instructions", "known_info", "unknown_info"):
            v = ins.get(k)
            if v:
                parts.append(f"{k}: {v}")
    elif isinstance(ins, str):
        parts.append(ins)
    if not parts:
        parts.append(str(raw.get("description") or raw)[:2000])
    user_msg = "\n".join(parts)
    return (
        f"You are a customer-support agent for the {task['domain']} domain.\n"
        f"Read the user's request and respond directly.\n\n"
        f"User scenario:\n{user_msg}\n"
    )
