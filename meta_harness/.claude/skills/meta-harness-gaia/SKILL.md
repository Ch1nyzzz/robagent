---
name: meta-harness-gaia
description: Run one iteration of agent harness evolution for GAIA.
---

# Meta-Harness (GAIA)

Run ONE iteration of agent scaffold evolution against the GAIA validation benchmark.

**You do NOT run benchmarks.** You analyze prior results + failed trajectories, propose ONE agent variant, implement it, and write `pending_eval.json`. The outer loop (`meta_harness/meta_harness.py`) handles benchmarking on the train-30 subset.

## CRITICAL CONSTRAINTS

- You MUST produce exactly 1 new agent variant every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- Do NOT touch `bench/gaia/scorer.py` or `evals.lock` (CI-protected eval contract).
- Do NOT modify `agent/base.py`, any existing `agent/v{N}/`, or any earlier `agent/mh_*/`.
- Do NOT modify `meta_harness/meta_harness.py`, `meta_harness/claude_wrapper.py`, or `run_benchmark.py`.

## Anti-overfitting rules

- **No task-specific hints.** Do not hardcode knowledge about specific GAIA tasks (entity names like "Finding Nemo", "USGS", artist names, paper titles, ...).
- **Never mention task names** in agent code, prompts, or comments. No `if "wikipedia" in question and "nedoshivina" in question:` patterns.
- **General guidance is OK.** Rules like "always strip trailing punctuation from the answer" or "if the question contains a URL, fetch it before answering" are fine — they happen to help specific tasks but apply broadly. Test: would this advice be useful to a researcher reading ANY unfamiliar Q&A task?
- **If in doubt, make it more general.** "Always extract the last span after `Answer:`" > "For movie title questions, look for an italicized phrase."

## CONTEXT

You are evolving the **base agent** at `agent/base.py`. It is a single LLM call with a tight system prompt and `max_tokens=2048`. The shared LLM client is at `agent/llm.py` (Together AI, DeepSeek-V4-Pro), and the event log lives in `agent/events.py`.

**Search space**: arbitrary Python code in your candidate directory. You can:

- Make multiple LLM calls (planning, retrieval, verification).
- Add tools (web fetch, file reading, calculators, retrieval).
- Add deterministic post-processing (answer extraction, normalization).
- Restructure into a state machine.

**Hard interface contract** (do not deviate):

Your file `agent/<candidate_name>/base.py` must export:

```python
def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # MUST return a dict with at least:
    #   - "answer": str | None  (the final answer string passed to the scorer)
    #   - "trace_path": str     (path to the per-task JSONL event log)
    # SHOULD also return:
    #   - "log": EventLog       (so run_benchmark can emit eval.scored on the same log)
    #   - "root_event_id": str  (for parent linkage on the eval event)
    #   - "error": str | None
```

See `agent/base.py` for a 30-line reference implementation that satisfies this contract. **Copy that file as your starting point** and modify from there.

`extras` always contains `{"level": "1"|"2"|"3", "file_name": "" or filename}`.

## CANDIDATE DESIGN

Each candidate lives at `agent/<candidate_name>/base.py` plus optional sibling files. The directory name MUST start with `mh_iter<N>_` and match `^mh_iter\d+_[a-z0-9_]+$` (e.g. `mh_iter1_longer_completion`).

You also need an empty `agent/<candidate_name>/__init__.py` so Python treats it as a package.

### Design principles

- **One mechanism per candidate.** Each candidate tests exactly one hypothesis (e.g. "the empty-completion / truncation failures will drop if max_tokens=8192 and we add a strict answer-extraction regex"). If you're tempted to add "and also a retrieval tool" — that's iter N+1.
- **Mechanism-first.** Identify a specific failure mode from the trace inventory below, then design changes that target it. Never add changes speculatively.
- **Stay within the LLM budget.** Don't fan out 20 LLM calls per task; the project uses a 60 qpm rate limit (see `agent/llm.py`).
- **Reuse `agent/events.py` and `agent/llm.py`.** Don't reinvent rate-limiting or trace logging.

## WORKFLOW (you, the proposer, do these steps)

### Step 1 — Read state

Read in this order. The exact file paths are in your runtime prompt; do not invent them.

1. `meta_harness/logs/frontier_val.json` — per-task best agent + score on train-30
2. `meta_harness/logs/evolution_summary.jsonl` — every prior candidate's hypothesis, score, delta
3. `agent/base.py` (the v0 reference; you must understand this before modifying)
4. The last 1-3 prior candidate dirs `agent/mh_iter*_*/base.py` if they exist (don't reread all)
5. **Failed trajectory sampling**: pick 4-6 task_ids from train-30 where the frontier is wrong (score=0). For each, read its newest trace at `traces/runs/gaia__<task_id>__*.jsonl`. Focus on `llm.responded.fields.finish_reason`, `llm.responded.fields.content`, and `answer.emitted.fields.answer`.

### Step 2 — Form ONE hypothesis

State in the form:

> **HYPOTHESIS**: "<falsifiable claim about what will improve train-30 score>"
>
> **MECHANISM**: <which observed failure mode this targets>, present in N≥3 distinct task_ids (list them)
>
> **PREDICTION**: train-30 acc will go from <current frontier acc> to roughly <X>

The mechanism MUST be observable in ≥3 train-30 traces. If you can't find 3, your hypothesis is too narrow — broaden it.

### Step 3 — Implement

1. Create `agent/<candidate_name>/__init__.py` (empty).
2. Create `agent/<candidate_name>/base.py`. Start by copying `agent/base.py` verbatim, then make targeted changes consistent with your hypothesis.
3. Keep `run_task`'s return-dict shape identical to `agent/base.py`.
4. Smoke-validate the import:

```bash
python -c "from agent.<candidate_name>.base import run_task; print('ok')"
```

If import fails, fix it before continuing.

### Step 4 — Write pending_eval.json

Write `meta_harness/logs/pending_eval.json` with:

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "<candidate_name>",
    "agent_version_arg": "<candidate_name>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<which failure this targets>",
    "evidence_task_ids": ["<id1>", "<id2>", "<id3>"],
    "changes": "<plain-English summary of what changed vs v0/prior frontier>",
    "expected_delta": "<expected acc change on train-30>"
  }
}
```

`agent_version_arg` is exactly what gets passed to `run_benchmark.py --agent-version`.

### Step 5 — Output

End your reply with a single line:

```
CANDIDATE: <candidate_name>
```

Nothing else after that line.

## EXAMPLE FAILURE PATTERNS YOU MIGHT TARGET

(For inspiration — verify the mechanism is actually present in the current traces before targeting it. Don't assume.)

- **Truncation**: `finish_reason=length` with empty / cut-off `content` → raise `max_tokens` AND/OR ask the model for the answer first then the reasoning.
- **Verbosity leakage**: model outputs "The answer is 42." instead of "42" → add a strict extractor over the response.
- **File ignored**: `extras.file_name` is non-empty but the agent never reads the file → route file-bearing tasks to a "block-and-emit-event" path (honest failure beats hallucination) OR add a minimal reader.
- **URL ignored**: question contains a URL but the agent has no fetcher → either fetch it or honestly block.
- **Number formatting**: ground truth is `1456` but answer is `1,456` or `1.456` → normalize before submission.

## STOP CONDITIONS (and what to do)

- If `agent/<candidate_name>/` already exists with content, your candidate_name collides. Pick a different name and try again.
- If you cannot find ≥3 traces matching your mechanism, pick a different mechanism.
- If you genuinely think the only remaining wins need new capabilities (vision, retrieval, file readers), pick the simplest one and implement it. The outer loop's job is to budget; yours is to propose.

## WHAT THIS SKILL DOES NOT DO

- Run benchmarks (the outer loop runs `run_benchmark.py`).
- Modify the eval rubric (`bench/gaia/scorer.py` is read-only).
- Touch v1-v13 (those are user-curated history; you're a parallel evolution branch starting from v0).
- Make claims about the held-out test-135. The outer loop will sweep the frontier candidate on test at the end of the run — your job is train-30.
