"""Compute claim-graph statistics for a versioned agent run.

Usage:
    python harness/gaia/_claim_graph_stats.py v10
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _latest_traces_for(marker: str) -> dict[str, str]:
    """Return latest trace per task_id where agent_version == marker."""
    files = sorted(
        glob.glob(str(ROOT / "traces" / "runs" / "gaia__*.jsonl")),
        key=lambda p: os.stat(p).st_mtime,
        reverse=True,
    )
    out: dict[str, str] = {}
    for fp in files:
        try:
            with open(fp) as f:
                first = f.readline()
            if not first.strip():
                continue
            rec = json.loads(first)
            if (rec.get("fields") or {}).get("agent_version") != marker:
                continue
            base = os.path.basename(fp)
            parts = base.split("__")
            if len(parts) < 3:
                continue
            tid = parts[1]
            out.setdefault(tid, fp)
        except Exception:
            continue
    return out


def main(marker: str) -> None:
    traces = _latest_traces_for(marker)
    print(f"version={marker}  traces={len(traces)}")

    route_counter: Counter = Counter()
    block_counter: Counter = Counter()
    tool_counter: Counter = Counter()
    n_with_claim = 0
    n_with_source = 0
    n_with_verified = 0
    n_unsourced_block = 0
    total_claims = 0
    total_sources = 0
    n_retrieval_planned = 0
    n_retrieval_empty = 0
    n_wiki_search_hit = 0
    n_wiki_search_called = 0
    n_file_reads_ok = 0
    n_file_reads_total = 0
    n_vision_ok = 0
    n_vision_total = 0
    answers_emitted = 0
    empty_answers = 0
    runs_with_error = 0

    for tid, fp in traces.items():
        events = []
        for line in open(fp):
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        if not events:
            continue
        by = {}
        for e in events:
            by.setdefault(e.get("type"), []).append(e)
        route = (by.get("task.routed", [{}])[0].get("fields") or {}).get("route")
        route_counter[str(route)] += 1
        for be in by.get("agent.blocked", []):
            block_counter[(be.get("fields") or {}).get("reason", "")] += 1
        for te in by.get("tool.called", []):
            tool_counter[(te.get("fields") or {}).get("tool", "")] += 1
        if by.get("claim.extracted"):
            n_with_claim += 1
            total_claims += len(by["claim.extracted"])
        if by.get("source.opened"):
            n_with_source += 1
            total_sources += len(by["source.opened"])
        for ve in by.get("claim.verified", []):
            if (ve.get("fields") or {}).get("ok"):
                n_with_verified += 1
                break
        for be in by.get("agent.blocked", []):
            reason = (be.get("fields") or {}).get("reason") or ""
            if reason.startswith("claim_unverified"):
                n_unsourced_block += 1
                break
        if by.get("retrieval.planned"):
            n_retrieval_planned += 1
        if by.get("retrieval.empty"):
            n_retrieval_empty += 1
        for te in by.get("tool.returned", []):
            f = te.get("fields") or {}
            tool = f.get("tool")
            ok = f.get("ok")
            if tool == "wikipedia_search":
                n_wiki_search_called += 1
                if (f.get("n_hits") or 0) > 0:
                    n_wiki_search_hit += 1
            elif tool == "read_gaia_file":
                n_file_reads_total += 1
                if ok:
                    n_file_reads_ok += 1
            elif tool == "vision_describe":
                n_vision_total += 1
                if ok:
                    n_vision_ok += 1
        for ae in by.get("answer.emitted", []):
            answers_emitted += 1
            if not (ae.get("fields") or {}).get("answer", "").strip():
                empty_answers += 1
        if by.get("run.failed"):
            runs_with_error += 1

    print(f"routes: {dict(route_counter)}")
    print(f"tools called: {dict(tool_counter)}")
    print(f"blocks: {dict(block_counter)}")
    print(f"tasks with claim.extracted: {n_with_claim}")
    print(f"tasks with source.opened:   {n_with_source}")
    print(f"tasks with claim.verified.ok=True: {n_with_verified}")
    print(f"tasks with claim_unverified block: {n_unsourced_block}")
    print(f"total claims emitted:  {total_claims}")
    print(f"total sources opened:  {total_sources}")
    print(f"retrieval planned:     {n_retrieval_planned}")
    print(f"retrieval empty:       {n_retrieval_empty}")
    print(
        f"wiki search hit rate:  {n_wiki_search_hit}/{n_wiki_search_called}"
        f" = {n_wiki_search_hit/max(n_wiki_search_called,1):.2f}"
    )
    print(
        f"file read success:     {n_file_reads_ok}/{n_file_reads_total}"
        f" = {n_file_reads_ok/max(n_file_reads_total,1):.2f}"
    )
    print(
        f"vision success:        {n_vision_ok}/{n_vision_total}"
        f" = {n_vision_ok/max(n_vision_total,1):.2f}"
    )
    print(f"answers emitted:       {answers_emitted} (empty: {empty_answers})")
    print(f"runs with run.failed:  {runs_with_error}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v10")
