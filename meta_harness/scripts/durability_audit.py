"""Post-champion durability audit for the robust / component skills.

Two source modes:

  --source plugin (default)
    Original robust-skill mode. Each candidate is one plugin classified on
    the durability axis; conditional plugins emit
    `plugin.activated` / `plugin.inert` markers per task. The audit scans
    the trace files referenced by a run summary jsonl, counts activated
    vs inert per plugin, and flags INERT / SUSPECT / LIVE.

  --source component
    New component-harness-tau2 mode. Each candidate is one workflow node;
    the runtime appends one row to `.component-state/<tag>/fired.jsonl`
    per fire (fields: ts / component / mount / decision / ...). This mode
    counts fires per component, cross-references the run summary for
    per-task reward, and flags:

      INERT    component never fired across the run
      SUSPECT  fires > 0 but never on a task with score > 0
      LIVE     fires > 0 with >= 1 correct-task fire

    The trust manifest (evidence_anchor, blast_radius, rollback_when,
    out_of_evidence_probe) is read from evolution_summary.jsonl row's
    `plugin.component.trust` block.

Usage:
  # robust / GAIA (original)
  python meta_harness/scripts/durability_audit.py \
      --logs-dir meta_harness/logs_robust \
      --summary-path traces/gaia_<agent>__summary.jsonl

  # component / tau2
  python meta_harness/scripts/durability_audit.py --source component \
      --logs-dir meta_harness/logs_tau2_components \
      --fired meta_harness/.component-state/iter5/fired.jsonl \
      --summary-path traces/tau2_component_runtime__summary.jsonl
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


def load_component_manifests(evolution_summary: Path) -> dict[str, dict]:
    """Collect component manifests from a component-harness evolution_summary.

    The `plugin` key in each row holds the combined component +
    workflow_patch payload from meta_harness_components.py. We index by
    `component.id` (or fall back to workflow_patch.name for disable_node).
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
        payload = row.get("plugin")
        if not isinstance(payload, dict):
            continue
        comp = payload.get("component") or {}
        wf = payload.get("workflow_patch") or {}
        name = comp.get("id") or wf.get("name")
        if name:
            manifests[name] = {
                "cls": comp.get("cls"),
                "mount": comp.get("mount"),
                "trust": comp.get("trust") or {},
                "workflow_op": wf.get("op"),
            }
    return manifests


def load_component_fires(fired_path: Path) -> list[dict]:
    """Read .component-state/<tag>/fired.jsonl. One row per fire."""
    rows: list[dict] = []
    if not fired_path.exists():
        return rows
    for line in fired_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


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


def audit_components(manifests: dict[str, dict], fires: list[dict],
                     run: list[dict]) -> list[dict]:
    """Component-mode audit.

    Counts fires per component, cross-references run summary for
    per-task reward, and flags INERT / SUSPECT / LIVE. fired.jsonl rows
    do not carry task_id today, so the "correct-when-fired" measure
    aggregates: a component is LIVE iff it fires at all in a run that
    has >=1 correct task. (When the runtime grows per-task tags, this
    tightens to per-task correctness.)
    """
    fires_by_name = defaultdict(int)
    decisions_by_name = defaultdict(lambda: defaultdict(int))
    for f in fires:
        name = f.get("component")
        if not name:
            continue
        fires_by_name[name] += 1
        d = f.get("decision")
        if d:
            decisions_by_name[name][d] += 1

    correct_tasks = sum(1 for r in run if r["score"] > 0)
    all_names = set(fires_by_name) | set(manifests)

    report: list[dict] = []
    for name in sorted(all_names):
        fires_n = fires_by_name[name]
        if fires_n == 0:
            flag = "INERT"
        elif correct_tasks == 0:
            flag = "SUSPECT"
        else:
            flag = "LIVE"
        m = manifests.get(name, {})
        trust = m.get("trust") or {}
        report.append({
            "name": name,
            "cls": m.get("cls", "?"),
            "mount": m.get("mount", "?"),
            "fires": fires_n,
            "decision_distribution": dict(decisions_by_name[name]),
            "flag": flag,
            "evidence_anchor": trust.get("evidence_anchor", ""),
            "blast_radius": trust.get("blast_radius", ""),
            "rollback_when": trust.get("rollback_when", ""),
            "out_of_evidence_probe": trust.get("out_of_evidence_probe", ""),
            "has_manifest": name in manifests,
        })
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["plugin", "component"], default="plugin",
                   help="audit source: 'plugin' (robust/GAIA, scans traces "
                        "for plugin.activated markers) or 'component' "
                        "(component-harness-tau2, reads .component-state fired.jsonl)")
    p.add_argument("--logs-dir", type=Path, required=True,
                   help="run state dir holding evolution_summary.jsonl")
    p.add_argument("--summary-path", type=Path, default=None,
                   help="run summary jsonl to audit (e.g. a test-135 run)")
    p.add_argument("--agent", default=None,
                   help="agent name; resolves --summary-path to "
                        "traces/gaia_<agent>__summary.jsonl (plugin mode) when omitted")
    p.add_argument("--fired", type=Path, default=None,
                   help="component mode: path to .component-state/<tag>/fired.jsonl")
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = p.parse_args()

    summary_path = args.summary_path
    if summary_path is None:
        if not args.agent:
            raise SystemExit("provide --summary-path or --agent")
        summary_path = ROOT / "traces" / f"gaia_{args.agent}__summary.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    if args.source == "plugin":
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
            print(f"  INERT (dead weight): {', '.join(inert)}")
        if suspect:
            print(f"  SUSPECT (fires but never on a correct task): {', '.join(suspect)}")
        if not inert and not suspect:
            print("  no INERT or SUSPECT plugins — every instrumented plugin is LIVE.")
        return

    # component mode
    if args.fired is None:
        raise SystemExit("--source component requires --fired PATH")
    if not args.fired.exists():
        raise SystemExit(f"fired.jsonl not found: {args.fired}")

    manifests = load_component_manifests(args.logs_dir / "evolution_summary.jsonl")
    fires = load_component_fires(args.fired)
    run = load_run(summary_path)
    report = audit_components(manifests, fires, run)

    if args.json:
        print(json.dumps({
            "summary_path": str(summary_path),
            "fired_path": str(args.fired),
            "tasks": len(run),
            "components": report,
        }, indent=2))
        return

    print(f"\nComponent durability audit — {args.fired.name}")
    print(f"  run: {summary_path.name}  ({len(run)} tasks, "
          f"{sum(1 for r in run if r['score']>0)} correct)")
    if not report:
        print("  no component fires and no manifests. Nothing to audit.")
        return
    print(f"  {'component':<40} {'cls':<18} {'mount':<20} {'fires':>6}  flag")
    print("  " + "-" * 100)
    for r in report:
        manifest_mark = "" if r["has_manifest"] else "  (no manifest)"
        print(f"  {r['name']:<40} {r['cls']:<18} {r['mount']:<20} "
              f"{r['fires']:>6}  {r['flag']}{manifest_mark}")

    inert = [r["name"] for r in report if r["flag"] == "INERT"]
    suspect = [r["name"] for r in report if r["flag"] == "SUSPECT"]
    print()
    if inert:
        print(f"  INERT (dead weight): {', '.join(inert)}")
    if suspect:
        print(f"  SUSPECT (fires but no correct task in this run): {', '.join(suspect)}")
    if not inert and not suspect:
        print("  no INERT or SUSPECT components — every instrumented component is LIVE.")


if __name__ == "__main__":
    main()
