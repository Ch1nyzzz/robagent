"""End-to-end smoke test: SopBenchAgent v0 on 1 task per domain.

Validates that:
  - chat() with tools= works against the locked target model
  - Bedrock toolspec -> OpenAI tool format conversion is accepted upstream
  - ToolManager.execute_tool dispatch works inside the agent loop
  - evaluate() can parse SopBenchAgent's AgentResult into metrics

Run:
    set -a && source /data/home/yuhan/robagent/.env && set +a && \
    python meta_harness/scripts/sopbench_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amazon_sop_bench import evaluate  # noqa: E402

from agent.sopbench_agent import SopBenchAgent  # noqa: E402

import os

# Default smoke covers all 3 SOP-Bench targets; override with SOPBENCH_SMOKE_DOMAINS
# (comma-separated) to re-run a subset (e.g., after a fix on the printing side).
DOMAINS = [
    d.strip()
    for d in os.environ.get(
        "SOPBENCH_SMOKE_DOMAINS",
        "dangerous_goods,warehouse_package_inspection,traffic_spoofing_detection",
    ).split(",")
    if d.strip()
]


def main() -> None:
    for domain in DOMAINS:
        print(f"\n{'=' * 60}\n=== smoke: {domain} (1 task)\n{'=' * 60}")
        agent = SopBenchAgent(max_iterations=10, max_tokens=8192)
        results = evaluate(
            benchmark_name=domain,
            agent=agent,
            max_tasks=1,
            max_workers=1,
        )
        print(f"benchmark      : {results.get('benchmark_name', domain)}")
        print(f"num_tasks      : {results.get('num_tasks')}")
        print(f"num_correct    : {results.get('num_correct')}")
        print(f"TSR            : {float(results.get('task_success_rate', 0)):.1%}")
        print(f"ECR            : {float(results.get('execution_completion_rate', 0)):.1%}")
        ta = results.get("tool_accuracy", 0)
        if isinstance(ta, dict):
            overall = ta.get("Overall") if "Overall" in ta else ta.get("overall", 0)
            print(f"Tool Accuracy  : overall={float(overall or 0):.1%}  by_tool={ta}")
        else:
            print(f"Tool Accuracy  : {float(ta or 0):.1%}")
        trs = results.get("task_results") or []
        if trs:
            tr = trs[0]
            print(f"first task     : {getattr(tr, 'task_id', '?')}  "
                  f"success={getattr(tr, 'success', '?')}  "
                  f"error={getattr(tr, 'error', None)}")


if __name__ == "__main__":
    main()
