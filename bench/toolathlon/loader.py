"""Toolathlon task loader.

Iterates the vendored `Toolathlon-src/tasks/finalpool/` directory and yields
one dict per task. Unlike GAIA (HF dataset) or tau2 (Python sim with task
JSON), Toolathlon tasks are directories on disk: each contains
`task_config.json`, `docs/`, `initial_workspace/`, `groundtruth_workspace/`,
and `evaluation/main.py`.

A task is a single Python interaction with a containerized environment, so
we do NOT eagerly load its workspace files; we just hand the task_id +
`task_dir` to the runner.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator


def _toolathlon_root() -> Path:
    root = os.environ.get("TOOLATHLON_ROOT")
    if root:
        p = Path(root)
        if p.exists():
            return p
    here = Path(__file__).resolve().parent.parent.parent
    candidate = here / "Toolathlon-src"
    if candidate.exists():
        return candidate
    raise RuntimeError(
        "Cannot locate Toolathlon source. Set TOOLATHLON_ROOT or "
        "clone hkust-nlp/Toolathlon into ./Toolathlon-src."
    )


# Credential-tier classification — keep in sync with the analysis in the
# integration log. A task's tier is the max of its needed_mcp_servers' tiers.
#
#   LV0a: zero external credentials.
#   LV0b: Serper API key (web_search local tool needs it).
#   LV1:  local docker apps via deploy_containers.sh (poste/canvas/woo/kind).
#   LV2:  GitHub PAT or WandB account.
#   LV3:  GCP / Notion / Snowflake — paid / heavy setup.
MCP_TIER = {
    # LV1 — local docker apps
    "emails": "LV1",
    "canvas": "LV1",
    "woocommerce": "LV1",
    "k8s": "LV1",
    # LV2 — free remote SaaS
    "github": "LV2",
    "wandb": "LV2",
    # LV3 — paid / heavy remote SaaS
    "google-cloud": "LV3",
    "google_sheet": "LV3",
    "google_calendar": "LV3",
    "google_forms": "LV3",
    "notion": "LV3",
    "snowflake": "LV3",
    # everything else defaults to LV0a (or LV0b via web_search override below)
}
_TIER_ORDER = {"LV0a": 0, "LV0b": 1, "LV1": 2, "LV2": 3, "LV3": 4}


def _classify_tier(mcps: list[str], local_tools: list[str]) -> str:
    tier = "LV0a"
    if "web_search" in local_tools:
        tier = "LV0b"
    for m in mcps:
        t = MCP_TIER.get(m, "LV0a")
        if _TIER_ORDER[t] > _TIER_ORDER[tier]:
            tier = t
    return tier


def list_task_ids(tier_max: str = "LV3") -> list[str]:
    """Return all task ids whose tier <= tier_max (e.g. 'LV0b')."""
    tasks_dir = _toolathlon_root() / "tasks" / "finalpool"
    cap = _TIER_ORDER[tier_max]
    out: list[str] = []
    for d in sorted(tasks_dir.iterdir()):
        cfg = d / "task_config.json"
        if not cfg.exists():
            continue
        c = json.loads(cfg.read_text())
        t = _classify_tier(c.get("needed_mcp_servers", []) or [],
                           c.get("needed_local_tools", []) or [])
        if _TIER_ORDER[t] <= cap:
            out.append(d.name)
    return out


def iter_tasks(
    task_ids: list[str] | None = None,
    tier_max: str = "LV3",
) -> Iterator[dict]:
    """Yield Toolathlon tasks.

    Each yielded dict carries:
      task_id     — bare directory name (e.g. 'find-alita-paper')
      task_dir    — 'finalpool/<task_id>' (the path arg run_single_decoupled
                    expects)
      tier        — 'LV0a' | 'LV0b' | 'LV1' | 'LV2' | 'LV3'
      mcps        — needed_mcp_servers list
      local_tools — needed_local_tools list
      task_root   — absolute Path to the task directory
    """
    tasks_dir = _toolathlon_root() / "tasks" / "finalpool"
    cap = _TIER_ORDER[tier_max]
    allow = set(task_ids) if task_ids else None
    for d in sorted(tasks_dir.iterdir()):
        if allow is not None and d.name not in allow:
            continue
        cfg = d / "task_config.json"
        if not cfg.exists():
            continue
        c = json.loads(cfg.read_text())
        mcps = c.get("needed_mcp_servers", []) or []
        ltools = c.get("needed_local_tools", []) or []
        tier = _classify_tier(mcps, ltools)
        if _TIER_ORDER[tier] > cap:
            continue
        yield {
            "task_id": d.name,
            "task_dir": f"finalpool/{d.name}",
            "tier": tier,
            "mcps": mcps,
            "local_tools": ltools,
            "task_root": d,
        }


def read_task_prompt(task: dict) -> str:
    """Best-effort plain-text task description from docs/task.md.

    This is NOT what the live Toolathlon agent loop sees (that uses
    SystemPrompts built by utils.data_structures.task_config); this is only
    for logging / human inspection.
    """
    md = task["task_root"] / "docs" / "task.md"
    return md.read_text(encoding="utf-8") if md.exists() else ""
