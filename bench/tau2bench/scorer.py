from __future__ import annotations

from typing import Any


def official_evaluator_available() -> bool:
    try:
        importlib_test()
        return True
    except Exception:
        return False


def importlib_test() -> None:
    import importlib

    for mod in [
        "tau2.evaluator",
        "tau2.simulator",
    ]:
        importlib.import_module(mod)


def score_baseline(task: dict, answer: str) -> dict[str, Any]:
    """Best-effort score for a single-turn baseline.

    Returns a partial-credit dict so we can collect signal even though the
    base agent is not a real multi-turn tool-using agent.
    """
    raw = task["raw"]
    ec = raw.get("evaluation_criteria") or {}
    expected_actions = (
        ec.get("actions")
        if isinstance(ec, dict)
        else None
    ) or raw.get("expected_actions") or raw.get("ground_truth_actions") or []
    answer_l = (answer or "").lower()

    hits = 0
    misses: list[str] = []
    for a in expected_actions:
        name = a.get("name") if isinstance(a, dict) else str(a)
        if name and name.lower() in answer_l:
            hits += 1
        elif name:
            misses.append(name)

    has_response = bool(answer and answer.strip())
    return {
        "official_run": False,
        "has_response": has_response,
        "expected_action_count": len(expected_actions),
        "matched_action_mentions": hits,
        "missed_action_names": misses,
        "score": float(hits) / len(expected_actions) if expected_actions else (1.0 if has_response else 0.0),
        "note": "baseline single-turn score; not the official tau2 simulator score",
    }
