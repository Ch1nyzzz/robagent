---
name: robust-harness-gaia
description: Run ONE iteration of GAIA harness evolution. Propose ONE candidate that shifts behavior out of the LLM into deterministic Python, classify it on the durability axis (channel / reactive_guard / deterministic_glue / predictive_heuristic), and write a plugin manifest to pending_eval.json. The outer loop runs benchmarks.
---

# robust-harness-gaia

Run ONE iteration of agent evolution against GAIA. **You do NOT run benchmarks.** You analyze prior results + failed traces, propose ONE candidate, classify it on the durability axis, implement it, write `pending_eval.json` (a plugin manifest), and exit. The outer loop (`meta_harness/meta_harness.py`) evaluates on train-30.

## First principles (in priority order)

1. **The LLM is the last resort.** Every iteration finds one place where the LLM is doing work a regex / parser / lookup / arithmetic / state transition could do — and moves it into deterministic Python. The end state is many small focused LLM nodes wrapped in deterministic glue, not one big LLM call.

2. **Tools bridge channel gaps, not reasoning gaps.** Add a tool when traces prove the LLM could finish if given external content it can't reach (file content, web fetch, search). Do NOT add a tool when the model itself can't reason over the would-be output (vision on a text-only model) — mark `BLOCKED{reason=model_capability_gap}` and never revisit.

3. **Honest blocks > silent fabrication.** When the agent cannot proceed, emit `agent.blocked{reason=...}` with a structured event. Calibration is part of correctness.

4. **Build for durability, not for this model's weaknesses.** A change that patches the *current* model's weakness becomes dead weight — or actively harmful — once the model improves. Every candidate is classified on the durability axis (below) so the system can tell durable capability apart from a disposable patch.

(Inspired by 12-factor-agents — github.com/humanlayer/12-factor-agents. Apply the spirit; ignore the letter.)

## The durability axis: classify your candidate

Your candidate IS one plugin: ONE new or modified component. Its **class** is *derived* from what its activation predicate tests — you do not assert it, the predicate decides.

| class | the component's `if` tests... | when the model gets stronger | risk |
|---|---|---|---|
| `channel` | task structure — `extras.file_name`, a post-cutoff date, a named source | stays needed: the model still cannot *reach* external content | low |
| `reactive_guard` | an **observed failure event** — `finish_reason=length`, empty content, `tool.failed` | self-disables: a model that no longer fails never triggers it → zero drag | low |
| `deterministic_glue` | nothing — always-on transform of data already in hand (normalizer, extractor, parser) | durable; harmless even when redundant | low |
| `predictive_heuristic` | the **prompt text** — a regex/keyword guess about how the model *will* behave | dangerous: it overrides the model on a prior and can turn **negative** | HIGH |

**Preference order: `channel` > `reactive_guard` > `deterministic_glue` > `predictive_heuristic`.** Always reach for a lower-risk class first.

### The `predictive_heuristic` gate

A `predictive_heuristic` is the only class that can make the harness *worse* as the model improves (e.g. routing a task the model would have answered correctly into a lossy search path). Propose one ONLY when ALL THREE hold — otherwise pick a different class:

1. **≥3 evidence traces** share the exact mechanism it targets.
2. **Non-destructive fallback**: when the heuristic mis-fires, the path it forces must not yield an answer *worse* than not firing. If a wrong activation can override correct model knowledge, set `destructive_fallback: true` and justify in the manifest why the gain still outweighs it.
3. It emits `plugin.activated` / `plugin.inert` (see below) so its real value stays measurable after a model change.

If the gate fails, downgrade: act on the *observed* failure (`reactive_guard`) instead of predicting it, or use a `deterministic_glue` extractor, or mark the task `BLOCKED`.

## Activation markers: `plugin.activated` / `plugin.inert`

Every **conditional** plugin (`channel`, `reactive_guard`, `predictive_heuristic`) must make its firing observable. At the branch point emit exactly one uniform marker per task:

- engages → `log.emit("plugin.activated", parent=root, plugin="<plugin name>")`
- reachable but does NOT engage → `log.emit("plugin.inert", parent=root, plugin="<plugin name>")`

Keep emitting any domain events too (`file.read`, `task.routed`, `tool.called`). `deterministic_glue` is always-on — no markers. `EventLog.emit` accepts any type string, so **no change to `agent/events.py` is needed.**

These markers make durability *passively measurable*: after a model upgrade, `durability_audit.py` flags a plugin that is `plugin.inert` on 100% of tasks as dead weight, and one that fires often while tasks still fail as a suspect.

## Hard rules

- Exactly ONE new candidate per invocation. Do not loop. Do not write "the frontier is optimal" or abort early.
- **You do NOT run benchmarks** — no `run_benchmark.py`, no `tools/eval.py`, no train-30 / unit-test evals. The outer loop scores your candidate. Your job is to propose.
- **No task-specific code.** No entity-name literals ("Finding Nemo", "USGS", paper titles). General rules ("if `extras.file_name` is set → route to BLOCKED", "strip trailing period from numeric answers") are fine. A `predictive_heuristic` regex tuned to match a handful of specific train task_ids is task-specific code in disguise — the gate above exists to catch exactly that.
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
meta_harness/logs/evolution_summary.jsonl  every prior candidate + score + plugin manifest
meta_harness/train_task_ids.txt            30 tasks — your only sampling pool
```

(The exact paths are in your runtime prompt — use those, not these placeholders.)

Pick 4-6 train-30 tasks where the frontier scores 0. Read their newest trace at `traces/runs/gaia__<task_id>__*.jsonl`. Inspect `llm.responded.finish_reason`, `answer.emitted.answer`, `agent.blocked.reason`, `tool.called`, `claim.extracted`, `plugin.activated`, `plugin.inert`.

### 2. Form ONE hypothesis

> **HYPOTHESIS**: <falsifiable claim about train-30 score>
> **MECHANISM**: <failure mode> seen in N≥3 task_ids [tid1, tid2, ...]
> **FIX**: <what code moves out of LLM into Python>
> **CLASS**: <channel | reactive_guard | deterministic_glue | predictive_heuristic> — derived from the activation predicate
> **PREDICTION**: train-30 acc <current> → <expected>

If you can't find ≥3 traces sharing the mechanism, broaden it. Single-task fixes are rejected.

### 3. Classify (subsumes the capability check)

Determine the **class** from the activation predicate (see the durability axis). This replaces the old binary tool check:

- Feeds the LLM external content it cannot reach (file, web) → `channel`. Ask first: *"if a perfect channel gave the LLM exactly this content, would it finish correctly?"* If **no** (the model cannot reason over the content — e.g. vision on a text-only model) → do NOT add the channel; mark `BLOCKED{reason=model_capability_gap}`.
- Re-invokes / redirects the LLM after an observed failure event → `reactive_guard`.
- Pure deterministic pre/post-processing of data already in hand → `deterministic_glue`.
- Routes / rewrites based on a guess from the prompt → `predictive_heuristic`. Apply the §gate. If it fails, downgrade to `reactive_guard` or `deterministic_glue`.

Confirmed `model_capability_gap`s (do not retry): vision via Together — Llama-Vision needs a paid endpoint.

### 4. Implement

Create `agent/<candidate>/{__init__.py, base.py, ...}`. Keep `run_task`'s return-dict shape identical to `agent/base.py`. For a conditional plugin, emit `plugin.activated` / `plugin.inert` at its branch point. Validate the import:

```bash
python -c "from agent.<candidate>.base import run_task; print('ok')"
```

Fix any import error before continuing. This is the ONLY shell command you run — no benchmarks.

### 5. Write `pending_eval.json` (plugin manifest)

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "<candidate>",
    "agent_version_arg": "<candidate>",
    "hypothesis": "<falsifiable claim about train-30 score>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected acc change on train-30>",
    "plugin": {
      "name": "<stable component id — REUSE the existing id if you modify an existing component>",
      "class": "channel | reactive_guard | deterministic_glue | predictive_heuristic",
      "activation_predicate": "<the exact condition the component branches on>",
      "activation_event": "<domain event emitted when it engages, e.g. file.read — documentation>",
      "destructive_fallback": <bool>,
      "fallback": "<what happens when the component does not engage / mis-fires>",
      "dead_when": "<observable condition under which this component is provably dead weight>",
      "predictive_justification": "<REQUIRED iff class == predictive_heuristic: how all 3 gate conditions are met; omit otherwise>"
    }
  }
}
```

`agent_version_arg` is exactly what gets passed to `run_benchmark.py --agent-version` by the outer loop. The `plugin` block is consumed only by the robust pipeline — the outer loop reads `name` / `agent_version_arg` / `hypothesis` / `changes` exactly as before, so this manifest stays compatible.

### 6. Session log + exit

Write a concise session log to `meta_harness/logs/builder_sessions/iter<N>/log.md` — mechanism, hypothesis, plugin class + why, files written, any §3 capability decision. No full traces.

Final line of your reply:

```
CANDIDATE: <candidate>
```

(or `NO_CANDIDATE: <reason>` only if genuinely no mechanism is actionable — rare.)

## Common patterns (with their class)

- `finish_reason=length` empty content → bump `max_tokens` OR add a recovery pass keyed on the length event → `reactive_guard`.
- Verbose explanation wrapping the answer → strict extractor (regex on last `Answer:` line) → `deterministic_glue`.
- `extras.file_name` set, no `file.read` event → add a deterministic file reader → `channel` (or `BLOCKED` if binary/vision).
- Source named in the question, no `source.opened` event → web fetch keyed on the named source/URL → `channel`.
- Number / list / case format drift → extend `normalize_answer` → `deterministic_glue`.
- LLM emits "UNKNOWN" → `is_unknown_response` filter → honest `BLOCKED` → `deterministic_glue`.
- A regex over the prompt that guesses "the model will hallucinate here, force search" → `predictive_heuristic` — apply the gate; prefer routing on a named-source signal (`channel`) instead.

Verify the mechanism is in current traces. Don't assume.

## What this skill does NOT do

- Run benchmarks (the outer loop runs `run_benchmark.py` on train-30).
- Modify the eval rubric (`bench/gaia/scorer.py` is read-only).
- Modify prior agent versions, `run_benchmark.py`, `agent/llm.py`, `agent/events.py`, or `meta_harness/`.
- Loop or propose multiple candidates in one invocation.
- Add capabilities the LLM cannot use (§3 catches this).
- Propose a `predictive_heuristic` that fails the gate.
- Make strategic stop decisions — the outer loop + human handle that.
