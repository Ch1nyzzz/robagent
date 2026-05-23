"""Post-champion durability audit for the robust skill.

The robust skill classifies every candidate as one plugin on a durability axis
(channel / reactive_guard / deterministic_glue / predictive_heuristic) and makes
conditional plugins emit `plugin.activated` / `plugin.inert` markers per task.

This script is the *passive*, free durability signal — it never re-runs a
benchmark. It reads the plugin manifests from an evolution_summary.jsonl and
scans an agent run's traces to measure, per plugin:

  - activation rate          : activated / (activated + inert)
  - correct-when-activated   : of the activated tasks, how many the run got right

and flags:

  INERT    activation rate 0  -> dead weight for this model; safe to remove
  SUSPECT  activates but 0 correct-when-activated -> may be harming the run
  LIVE     activates and contributes to correct answers

Re-run it after a model upgrade: a plugin that flips weak-model-LIVE to
strong-model-INERT is a stale assumption the stronger model has absorbed; a
`predictive_heuristic` that flips to SUSPECT is actively dragging and should be
deprecated.

Usage:
  # audit the champion of a robust run on whatever split it was last evaluated
  python meta_harness/scripts/durability_audit.py \
      --logs-dir meta_harness/logs_robust \
      --summary-path traces/gaia_mh_iter11_robust_wsig_extend__summary.jsonl

  # or let it resolve the summary path from an agent name
  python meta_harness/scripts/durability_audit.py \
      --logs-dir meta_harness/logs_robust --agent mh_iter11_robust_wsig_extend
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_manifests(evolution_summary: Path) -> dict[str, dict]:
    """Collect plugin manifests from an evolution_summary.jsonl.

    One plugin id may be (re)defined across several iterations; the latest
    definition wins. Rows with no `plugin` block (e.g. baseline skill or
    pre-manifest history) are skipped.
    """
    manifests: dict[str, dict] = {}
    if not evolution_summary.exists():
        return manifests
    for line in evolution_summary.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        plugin = row.get("plugin")
        if isinstance(plugin, dict) and plugin.get("name"):
            manifests[plugin["name"]] = plugin
    return manifests


def load_run(summary_path: Path) -> list[dict]:
    """Read a run summary jsonl into {task_id, score, trace_path} records."""
    rows: list[dict] = []
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append({
            "task_id": r.get("task_id"),
            "score": float(r.get("score") or 0),
            "trace_path": r.get("trace_path"),
        })
    return rows


def scan_trace(trace_path: str) -> tuple[set[str], set[str]]:
    """Return (activated plugin names, inert plugin names) for one trace."""
    activated: set[str] = set()
    inert: set[str] = set()
    p = Path(trace_path) if trace_path else None
    if not p or not p.exists():
        return activated, inert
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = (ev.get("fields") or {}).get("plugin")
        if not name:
            continue
        if ev.get("type") == "plugin.activated":
            activated.add(name)
        elif ev.get("type") == "plugin.inert":
            inert.add(name)
    return activated, inert


def audit(manifests: dict[str, dict], run: list[dict]) -> list[dict]:
    # per plugin: activated tasks, inert tasks, correct-when-activated
    act = defaultdict(int)
    inert = defaultdict(int)
    correct_act = defaultdict(int)
    seen: set[str] = set()

    for rec in run:
        a, i = scan_trace(rec["trace_path"])
        seen |= a | i
        for name in a:
            act[name] += 1
            if rec["score"] > 0:
                correct_act[name] += 1
        for name in i:
            inert[name] += 1

    report: list[dict] = []
    for name in sorted(seen | set(manifests)):
        a, i, ca = act[name], inert[name], correct_act[name]
        total = a + i
        rate = a / total if total else 0.0
        if total == 0:
            flag = "NO-MARKERS"  # plugin never reached, or markers not emitted
        elif a == 0:
            flag = "INERT"
        elif ca == 0:
            flag = "SUSPECT"
        else:
            flag = "LIVE"
        m = manifests.get(name, {})
        report.append({
            "name": name,
            "class": m.get("class", "?"),
            "activated": a,
            "inert": i,
            "activation_rate": round(rate, 3),
            "correct_when_activated": ca,
            "flag": flag,
            "destructive_fallback": m.get("destructive_fallback"),
            "has_manifest": name in manifests,
        })
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", type=Path, required=True,
                   help="robust run state dir holding evolution_summary.jsonl")
    p.add_argument("--summary-path", type=Path, default=None,
                   help="run summary jsonl to audit (e.g. a test-135 run)")
    p.add_argument("--agent", default=None,
                   help="agent name; resolves --summary-path to "
                        "traces/gaia_<agent>__summary.jsonl when that is omitted")
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = p.parse_args()

    summary_path = args.summary_path
    if summary_path is None:
        if not args.agent:
            raise SystemExit("provide --summary-path or --agent")
        summary_path = ROOT / "traces" / f"gaia_{args.agent}__summary.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    manifests = load_manifests(args.logs_dir / "evolution_summary.jsonl")
    run = load_run(summary_path)
    report = audit(manifests, run)

    if args.json:
        print(json.dumps({"summary_path": str(summary_path),
                          "tasks": len(run), "plugins": report}, indent=2))
        return

    print(f"\nDurability audit — {summary_path.name}  ({len(run)} tasks)")
    if not report:
        print("  no plugin markers found. Either this run predates the manifest "
              "schema, or no conditional plugins were instrumented.")
        return
    print(f"  {'plugin':<26} {'class':<20} {'act':>4} {'inert':>6} "
          f"{'rate':>6} {'ok/act':>7}  flag")
    print("  " + "-" * 86)
    for r in report:
        manifest_mark = "" if r["has_manifest"] else "  (no manifest)"
        print(f"  {r['name']:<26} {r['class']:<20} {r['activated']:>4} "
              f"{r['inert']:>6} {r['activation_rate']:>6.3f} "
              f"{r['correct_when_activated']:>3}/{r['activated']:<3} "
              f"{r['flag']}{manifest_mark}")

    inert = [r["name"] for r in report if r["flag"] == "INERT"]
    suspect = [r["name"] for r in report if r["flag"] == "SUSPECT"]
    print()
    if inert:
        print(f"  INERT (dead weight for this model — candidates to remove): "
              f"{', '.join(inert)}")
    if suspect:
        print(f"  SUSPECT (fires but never on a correct task — may be harmful): "
              f"{', '.join(suspect)}")
    if not inert and not suspect:
        print("  no INERT or SUSPECT plugins — every instrumented plugin is LIVE.")


if __name__ == "__main__":
    main()
