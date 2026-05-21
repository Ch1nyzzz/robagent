---
name: robust-harness-gaia
description: Run ONE iteration of GAIA harness evolution. Propose ONE candidate that shifts behavior out of the LLM and into deterministic Python — helpers, normalizers, routers, or tools as structured channels. You analyze prior results + failed traces and write pending_eval.json; the outer loop runs benchmarks.
---

# robust-harness-gaia

Run ONE iteration of agent evolution against GAIA. **You do NOT run benchmarks.** You analyze prior results + failed traces, propose ONE candidate, implement it, write `pending_eval.json`, and exit. The outer loop (`meta_harness/meta_harness.py`) evaluates on train-30.

## First principles (in priority order)

1. **The LLM is the last resort.** Every iteration finds one place where the LLM is doing work a regex / parser / lookup / arithmetic / state transition could do — and moves it into deterministic Python. The end state is many small focused LLM nodes wrapped in deterministic glue, not one big LLM call.

2. **Tools bridge channel gaps, not reasoning gaps.** Add a tool when traces prove the LLM could finish if given external content it can't reach (file content, web fetch, search). Do NOT add a tool when the model itself can't reason over the would-be output (vision on a text-only model) — mark `BLOCKED{reason=model_capability_gap}` and never revisit.

3. **Honest blocks > silent fabrication.** When the agent cannot proceed, emit `agent.blocked{reason=...}` with a structured event. Calibration is part of correctness.

(Inspired by 12-factor-agents — github.com/humanlayer/12-factor-agents. Apply the spirit; ignore the letter.)

## Hard rules

- Exactly ONE new candidate per invocation. Do not loop. Do not write "the frontier is optimal" or abort early.
- **You do NOT run benchmarks** — no `run_benchmark.py`, no `tools/eval.py`, no train-30 / unit-test evals. The outer loop scores your candidate. Your job is to propose.
- **No task-specific code.** No entity-name literals ("Finding Nemo", "USGS", paper titles). General rules ("if `extras.file_name` is set → route to BLOCKED", "strip trailing period from numeric answers") are fine.
- READ-ONLY: `bench/gaia/scorer.py`, `evals.lock`, `agent/base.py`, all existing `agent/v*/` and `agent/mh_iter*/`, `run_benchmark.py`, `agent/llm.py`, `meta_harness/*.py`.
- Reuse `agent/llm.py` (rate-limited Together client) and `agent/events.py` (event log). Don't reinvent.

## Candidate interface

```python
# agent/<candidate>/base.py
def run_task(*, benchmark: str, task_id: str, task_prompt: str,
             extras: dict | None = None) -> dict:
    # MUST return: {"answer": str|None, "trace_path": str, ...}
    # SHOULD return: {"log": EventLog, "root_event_id": str, "error": str|None}
```

`<candidate>` matches `^mh_iter\d+_[a-z0-9_]+$`. Add an empty `__init__.py`. `extras` always contains `{"level": "1"|"2"|"3", "file_name": "..."}`.

Start by copying the prior frontier candidate's `base.py`; make ONE targeted change.

## Workflow

### 1. Read state

```
meta_harness/logs/frontier_val.json        per-task best across all candidates
meta_harness/logs/evolution_summary.jsonl  every prior candidate + score + delta
meta_harness/train_task_ids.txt            30 tasks — your only sampling pool
```

(The exact paths are in your runtime prompt — use those, not these placeholders.)

Pick 4-6 train-30 tasks where the frontier scores 0. Read their newest trace at `traces/runs/gaia__<task_id>__*.jsonl`. Inspect `llm.responded.finish_reason`, `answer.emitted.answer`, `agent.blocked.reason`, `tool.called`, `claim.extracted`.

### 2. Form ONE hypothesis

> **HYPOTHESIS**: <falsifiable claim about train-30 score>
> **MECHANISM**: <failure mode> seen in N≥3 task_ids [tid1, tid2, ...]
> **DETERMINISTIC FIX**: <what code moves out of LLM into Python — helper / router / parser / state-machine split / new tool>
> **PREDICTION**: train-30 acc <current> → <expected>

If you can't find ≥3 traces sharing the mechanism, broaden it. Single-task fixes are rejected.

### 3. Capability check (only if adding a tool)

Tools bridge **channel gaps**, not **reasoning gaps**. Before adding a tool, ask: *"If a perfect tool gave the LLM exactly what it needs, would the LLM finish correctly?"*

- **Yes** → add it. Emit `tool.called` / `tool.returned` / `source.opened` events.
- **No** (model can't reason over the tool's output) → mark BLOCKED with `model_capability_gap`. Do not add the tool.
- **Unclear** → prefer the smaller change: skip the tool this iteration, target a deterministic helper instead. The outer loop's train-30 result will tell you next iteration whether the tool is worth it.

Confirmed `model_capability_gap`s (do not retry):
- vision via Together — Llama-Vision needs a paid endpoint.

### 4. Implement

Create `agent/<candidate>/{__init__.py, base.py, ...}`. Keep `run_task`'s return-dict shape identical to `agent/base.py`. Validate the import:

```bash
python -c "from agent.<candidate>.base import run_task; print('ok')"
```

Fix any import error before continuing. This is the ONLY shell command you run — no benchmarks.

### 5. Write `pending_eval.json`

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "<candidate>",
    "agent_version_arg": "<candidate>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure this targets>",
    "deterministic_fix": "<helper | router | parser | state_machine_split | new_tool>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "new_helper": <bool>, "new_tool": <bool>, "new_gate": <bool>,
    "expected_delta": "<expected acc change on train-30>"
  }
}
```

`agent_version_arg` is exactly what gets passed to `run_benchmark.py --agent-version` by the outer loop.

### 6. Session log + exit

Write a concise session log to `meta_harness/logs/builder_sessions/iter<N>/log.md` — mechanism, hypothesis, files written, any §3 capability decision. No full traces.

Final line of your reply:

```
CANDIDATE: <candidate>
```

(or `NO_CANDIDATE: <reason>` only if genuinely no mechanism is actionable — rare.)

## Common deterministic-fixable patterns

- `finish_reason=length` empty content → bump `max_tokens` OR split into plan+answer state-machine nodes.
- Verbose explanation wrapping the answer → strict extractor (regex on last `Answer:` line).
- `extras.file_name` set, no `file.read` event → route to BLOCKED OR add a deterministic file reader.
- Source mentioned in question, no `source.opened` event → router miss; extend signal regex.
- Number / list / case format drift → extend `normalize_answer`.
- LLM emits "UNKNOWN" → `is_unknown_response` filter → honest BLOCKED.

Verify the mechanism is in current traces. Don't assume.

## What this skill does NOT do

- Run benchmarks (the outer loop runs `run_benchmark.py` on train-30).
- Modify the eval rubric (`bench/gaia/scorer.py` is read-only).
- Modify prior agent versions, `run_benchmark.py`, `agent/llm.py`, or `meta_harness/`.
- Loop or propose multiple candidates in one invocation.
- Add capabilities the LLM cannot use (§3 catches this).
- Make strategic stop decisions — the outer loop + human handle that.
