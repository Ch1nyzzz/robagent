---
name: component-harness-gaia
description: Propose ONE hook (a single Python file declaring `COMPONENT: Component`) that stabilizes the GAIA main agent on a recurring failure mode, plus a workflow patch (add / replace / disable) against `meta_harness/workflows/gaia_main.yaml`. The main agent runs a tool-using FC loop (file_read / url_fetch / web_search / python_exec); hooks subscribe to lifecycle events via `listens=` and react via a Decision (allow / block / rewrite / inject_context). Admission gated by a class×event×decision matrix and a Trust block (evidence_anchor + out_of_evidence_probe).
---

# component-harness-gaia

The GAIA main agent is the protagonist. It runs a tool-using FC loop in `agent/base.py` and `agent/component_runtime/base.py` over the locked SUT model with four tools (`file_read`, `url_fetch`, `web_search`, `python_exec`). **Your job is to stabilize it**, not replace it. You propose ONE hook — a single Python file under `agent/components/<name>.py` exporting `COMPONENT: Component` — plus a workflow patch (`add` / `replace` / `disable`). **You do NOT run benchmarks.** You read prior traces, write one hook, write `pending_eval.json`, exit. The outer loop scores it on train-30 and admits or rejects.

## First principles

`RESULTS.md §4` validated these. Do not violate.

0. **Main agent is the protagonist.** Hooks stabilize its environment / outputs / retries; they don't replace its reasoning. Before writing a hook, ask: "would the main agent still be the same agent without this — just less prone to failing on X?" If the answer is "no, it'd be doing something fundamentally different", the hook is too heavy.
1. **Capability vs Stabilization — keep them separate.** If the main agent is failing because it can't *reach* some content (a new MCP server, a new file format, a new API), **register a tool** — don't write a hook to inject the content. Hooks are for stabilizing what the agent already can do: sanitize tool args, normalize outputs, recover from length truncation, etc.
2. **The LLM is the last resort within stabilization.** When you do write a hook, find one place the LLM is repeating deterministic work — output normalization, tool-arg shape validation, retry on observed failure — and move it into Python.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on a system field (`extras.file_name`), an LLM API field (`finish_reason`), a tool's JSON schema, a general algorithm. A hook INDUCED from N failed traces is memorisation. `RESULTS.md §4.5`: 10 layers at ε=5% compound to ~40% test false-positive.
4. **Honest over fabricated.** When the agent cannot satisfy the task within its reach, it says so (`agent.blocked`); never invent.

## Toolbox (pick what fits — none of these are exclusive)

| When you'd reach for it | What you change |
|---|---|
| Main agent can't reach something it needs (file type / API / KB) | Register a new tool in `agent/tools/` (Capability expansion — separate workflow, **not this skill**) |
| Main agent reaches it but mis-shapes the call / mis-uses the result | Write a `pre_tool_use` / `post_tool_use` hook (this skill) |
| LLM truncates / empties out on hard tasks | Write a `reactive_guard` on `on_length_truncation` / `on_empty_response` (this skill) |
| Final answer format drifts ("Five" vs "5", "$3.50" vs "3.50") | Write a `mechanism_layer` on `pre_answer_emit` (this skill) |
| Multiple hooks need to coordinate (A detects, B responds) | Custom event: A `ctx.emit("iter<N>_<slug>_X")`, B `listens="iter<N>_<slug>_X"` |

## The component model

```python
@dataclass(frozen=True, kw_only=True)
class Component:
    name: str                              # stable id; reuse for `replace`
    cls: ComponentClass                    # mechanism_layer | reactive_guard | induced_rule
    listens: str                           # event name the dispatcher routes on
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust                           # required verification block
    priority: int = 100                    # smaller fires first within an event bucket
    emits: tuple[str, ...] = ()            # self-doc of custom Tier-2/3 events this raises
```

### Lifecycle events the GAIA runtime emits

| event | when it fires | typical use |
|---|---|---|
| `task_received` | once per task; right after ctx built | static framework injection |
| `session_start` | once per task; paired with task_received | static framework injection |
| `pre_prompt_build` / `pre_context_build` | once per task; before initial messages built | rewrite the task prompt |
| `pre_agent_construct` | once per task; before messages sealed | inject inference hint |
| `pre_llm_request` | **per turn**; just before each `chat()` call | last-pass per-turn injection |
| `post_llm_response` / `post_llm_response_raw` | **per turn**; after each assistant turn | observe / rewrite raw_response |
| `on_length_truncation` | **per turn**; synthesised when `finish_reason=="length"` | sub-LLM recovery with bigger budget |
| `on_empty_response` | **per turn**; synthesised when content empty + no tool_calls | reactive retry |
| `pre_tool_use` | **per tool call**; before dispatch | sanitize `ctx.current_tool_args` (dict), or BLOCK to skip this call |
| `post_tool_use` | **per tool call**; after dispatch | rewrite `ctx.current_tool_result`, or INJECT to concatenate a note into it |
| `on_tool_error` | **per tool call**; synthesised when tool result starts with `ERROR:` | retry hint via inject_context |
| `pre_answer_emit` | once per task; after FC loop, before return | normalise / block (None=blocked) |
| `session_end` | bookkeeping | — |

### Component classes

| class | what the matcher tests | risk |
|---|---|---|
| `mechanism_layer` | system field / tool JSON Schema / LLM API field / file format / general algorithm | LOW |
| `reactive_guard` | observed failure event (`finish_reason=length`, empty content, tool error) | LOW |
| `induced_rule` | reading of policy/instruction text — IF/THEN from N evidence traces | HIGH (advisory `inject_context` only) |
| `predictive_heuristic` | raw prompt/response text via regex/keywords | REJECTED at load time |

### Decision

| decision | semantics |
|---|---|
| `allow` | no-op |
| `block` | terminate task; `answer=None` (EXCEPT at `pre_tool_use` where it skips just this tool call) |
| `rewrite` | replace the live payload at this event (see Event → payload below) |
| `inject_context` | setup events → append to `ctx.system_prompt`; post-LLM → queued as next-turn system note; `post_tool_use` / `on_tool_error` → concatenated INTO `ctx.current_tool_result` |

Event → REWRITE payload:

| event | payload type | replaces |
|---|---|---|
| `pre_prompt_build` / `pre_context_build` | `str` | `ctx.prompt` |
| `post_llm_response[_raw]` / `on_length_truncation` / `on_empty_response` | `str` | `ctx.raw_response` (this turn) |
| `pre_tool_use` | `dict` | `ctx.current_tool_args` |
| `post_tool_use` / `on_tool_error` | `str` | `ctx.current_tool_result` |
| `pre_answer_emit` | `str` or `None` | `ctx.answer` (None marks blocked) |

### Trust

Required: `evidence_anchor` / `blast_radius` (local|workflow|global) / `rollback_when`. `out_of_evidence_probe` is required for `induced_rule`. Optional: `fallback`.

- `evidence_anchor`: name a stable structure OUTSIDE your evidence — a system field, a tool's schema, an LLM API field, a general algorithm. If your anchor is "I observed trace_017/023/041 all do X", you're anchored INSIDE evidence — pick a different class or don't write the hook.
- `out_of_evidence_probe` (induced_rule only): name one concrete case NOT in evidence where matcher fires and what handler returns on it. If you can't, the rule overfits by construction.

### Handler helpers on `ctx`

| helper | what it does |
|---|---|
| `ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` | sub-LLM call via the locked SUT model |
| `ctx.shared` | per-task dict, free to read/write |
| `ctx.state[component_name]` | per-task per-component scratchpad |
| `ctx.emit("custom_event_name", **fields)` | fire custom event; re-enters dispatcher (depth cap 10) |
| `ctx.emit_upstream(key, value)` | shorthand for `ctx.upstream[key] = value` |

For FC-loop fields: `ctx.current_iter`, `ctx.current_tool_name`, `ctx.current_tool_args`, `ctx.current_tool_result`, `ctx.current_tool_call_id`, `ctx.current_tool_success` are populated only on per-turn / per-tool events.

## The class × event × decision matrix

Load-time gate at `agent/component_runtime/policy.py::ALLOWED`. A `(cls, listens)` not in the matrix → `ComponentPolicyError` at load. A handler returning a non-admitted Decision → raises at fire time.

| class \ event | setup (task_received / session_start / pre_prompt_build / pre_context_build / pre_agent_construct / pre_llm_request) | `post_llm_response[_raw]` / `on_length_truncation` / `on_empty_response` | `pre_tool_use` | `post_tool_use` | `on_tool_error` | `pre_answer_emit` | `session_end` |
|---|---|---|---|---|---|---|---|
| `mechanism_layer` | inject_context (+ rewrite/block on prompt_build) | rewrite, block, inject_context | rewrite, block, inject_context | rewrite, inject_context | inject_context | rewrite, block | allow |
| `reactive_guard` | — | rewrite, block, inject_context | rewrite, block, inject_context | rewrite, inject_context | inject_context | rewrite, block | — |
| `induced_rule` | **inject_context (advisory)** on prompt_build only | — | — | — | — | — | — |
| `predictive_heuristic` | rejected | rejected | rejected | rejected | rejected | rejected | rejected |

`induced_rule` stays advisory: an induced reading of policy as a `inject_context` hint is strictly safer than the LLM rediscovering the passage cold (worst case = redundant prompt). `predictive_heuristic` stays rejected: even advisory injection on a prompt-text regex conditions interpretation on raw surface text — structurally unsafe.

## The workflow file

```yaml
# meta_harness/workflows/gaia_main.yaml
nodes:
  - length_recovery_guard
  - cuneiform_numeric_decoder
disabled: []
```

Patch ops:

| op | meaning |
|---|---|
| `add` | append a new node; `agent/components/<id>.py` must be newly written |
| `replace` | keep the existing id; overwrite the file (same `COMPONENT.name`) |
| `disable` | move id into `disabled:`; file remains for the durability audit |

For `replace`, the **first shell action MUST be** `cp agent/components/<existing>.py agent/components/<existing>.py.bak_iter<N>` — the outer loop relies on the `.bak` to roll back on reject.

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_benchmark.py`, no `tools/eval.py`. The outer loop scores.
- **No task-specific code.** No entity names, no per-task branching, no encoded gold answers, no matchers keyed on `ctx.task_id`.
- **The target inference model is LOCKED.** `agent.llm.chat()` rejects any `model=` override. `ctx.chat()` does not accept a `model` kwarg.
- **Capability gaps are not for hooks.** If the main agent needs a new tool (new MCP, new file type, new API), file a separate request — do not invent a CHANNEL-style hook that injects content the agent should reach itself.
- READ-ONLY: `bench/`, `evals.lock`, `agent/base.py`, all `agent/v*/`, `agent/component_runtime/`, `agent/tools/`, all earlier `agent/components/*.py` (modify only via `replace`), `run_benchmark.py`, `agent/llm.py`, `agent/events.py`, all of `meta_harness/`.
- You may NOT create a new agent directory. Your only file writes are: ONE component file under `agent/components/<name>.py` (+ its `.bak_iter<N>` for `replace`) and the `pending_eval.json` manifest.

## Component file template

```python
# agent/components/component_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.extras / ctx.prompt / ctx.raw_response / ctx.answer /
    # ctx.shared / ctx.current_tool_* (on per-tool events) / ctx.current_iter.
    # NEVER read ctx.task_id.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)   # or inject_context / block / allow


COMPONENT = Component(
    name="<stable_component_id>",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="on_length_truncation",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="<system field / LLM API field / tool schema / general algorithm>",
        blast_radius="local",
        rollback_when="<observable rollback signal>",
        out_of_evidence_probe="<required for induced_rule; concrete OOE case + handler return>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

## Workflow

### 1. Read state

```
meta_harness/workflows/gaia_main.yaml                       active workflow
meta_harness/logs_components_gaia/frontier_workflow.json    frontier snapshot
meta_harness/logs_components_gaia/frontier_val.json         per-task best
meta_harness/logs_components_gaia/evolution_summary.jsonl   one row per iter (incl. rejected)
meta_harness/train_task_ids.txt                              30 train tasks
agent/components/                                           component files on disk
agent/tools/                                                 the baseline tools — read to know what's already covered
.component-state/iter<K>/fired.jsonl                        which components fired in iter K
traces/runs/iter<K>/gaia__<tid>__<run_id>.jsonl              per-task event log from iter K
```

Pick 4-6 train tasks the frontier still fails (`score == 0`). For each: read the trace. Note the **failure mechanism** (LLM lost the answer to length truncation? tool called with bad args? final answer in wrong format?). Skip tasks whose failure is "no tool exists for this kind of task" — that's a capability gap, not a stabilization problem.

### 2. Form ONE hypothesis

```
HYPOTHESIS:            <falsifiable claim about train-30 accuracy>
MECHANISM:             <failure mode> seen in N≥3 task traces [tid1, tid2, ...]
WHY STABILIZATION:     <why this is a stabilization gap, not a missing capability>
STABLE STRUCTURE:      <evidence_anchor — system field / tool schema / API field / algorithm;
                        must live OUTSIDE the N evidence traces>
OUT_OF_EVIDENCE PROBE: <required for induced_rule; concrete OOE case + handler return>
PATCH_OP:              <add | replace | disable>
COMPONENT:             listens=<...>, cls=<...>
EXPECTED_DELTA:        train-30 acc <current> → <expected>
```

If you can't name a STABLE STRUCTURE outside evidence, or your hook is essentially "give the agent content it can't otherwise reach" → re-examine. The latter is a tool request, not a hook.

### 3. Classify class + choose event

| matcher tests… | class | suggested events |
|---|---|---|
| LLM API field (`finish_reason=length`, empty content) | `reactive_guard` | `on_length_truncation` / `on_empty_response` |
| tool schema (drop unknown kwarg, fill missing required) | `mechanism_layer` | `pre_tool_use` (rewrite dict) |
| tool error string (4xx pattern, "not found") | `reactive_guard` | `on_tool_error` (inject retry hint) |
| post-tool result shape (truncate, summarize, flag) | `mechanism_layer` | `post_tool_use` (rewrite str / inject) |
| answer format (number normalize, strip prefixes, FINAL ANSWER regex) | `mechanism_layer` | `pre_answer_emit` (rewrite str) |
| framework constant ("output in X format") | `mechanism_layer` | `session_start` (inject_context) |
| policy/instruction reading (advisory) | `induced_rule` | `pre_prompt_build` (inject_context only) |
| raw prompt text via regex/keywords | none — redesign or don't write | — |

### 4. Implement + validate

Create exactly one file at `agent/components/<name>.py`. Validate:

```bash
python -c "
from agent.component_runtime.registry import load_components_from_dir
comps = load_components_from_dir(only=['<COMPONENT.name>'])
assert any(c.name == '<COMPONENT.name>' for c in comps)
print('loads + passes policy:', True)
"
```

For `replace`, FIRST run the `cp ... .bak_iter<N>` command. This is the ONLY shell command you run.

### 5. Write `pending_eval.json`

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "agent_version_arg": "component_runtime",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected acc change on train-30>",
    "component": {
      "id": "<stable_component_id>",
      "cls": "mechanism_layer | reactive_guard | induced_rule",
      "listens": "<event_name>",
      "file": "agent/components/<name>.py",
      "trust": {
        "evidence_anchor": "...",
        "blast_radius": "local | workflow | global",
        "rollback_when": "...",
        "out_of_evidence_probe": "<required for induced_rule>",
        "fallback": "..."
      }
    },
    "workflow_patch": {
      "op": "add | replace | disable",
      "name": "<component name>",
      "file": "agent/components/<name>.py"
    }
  }
}
```

Final line of your reply:

```
CANDIDATE: candidate_iter<N>_<slug>
```

## Common stabilization patterns

| pattern | class | listens | decision |
|---|---|---|---|
| length-recovery via sub-LLM with bigger budget | reactive_guard | `on_length_truncation` | rewrite (via `ctx.chat(max_tokens=32768)`) |
| recover from empty response (retry with hint) | reactive_guard | `on_empty_response` | rewrite or inject_context |
| strip `<thinking>` blocks from raw response | mechanism_layer | `post_llm_response` | rewrite |
| sanitize tool args (drop unknown kwargs / fill defaults) | mechanism_layer | `pre_tool_use` | rewrite (dict) |
| block a malformed / unsafe tool call | reactive_guard | `pre_tool_use` | block (skips this call only) |
| retry hint on tool error (e.g. URL 404 → "try wikipedia.org/wiki/X") | reactive_guard | `on_tool_error` | inject_context |
| summarize an oversized tool result before LLM sees it | mechanism_layer | `post_tool_use` | rewrite (str) |
| inject "remember the FINAL ANSWER format" framework note | mechanism_layer | `session_start` | inject_context |
| advisory: "if the answer is a date, use ISO 8601" | induced_rule | `pre_prompt_build` | inject_context |
| extract `FINAL ANSWER:` line / normalise number format | mechanism_layer | `pre_answer_emit` | rewrite |
| detect "I cannot answer" → BLOCKED | reactive_guard | `pre_answer_emit` | rewrite (to None) / block |
| two-hook coordination | mechanism_layer | A emits `iter<N>_<slug>_X` → B `listens="iter<N>_<slug>_X"` (reads `ctx.upstream`) | allow / inject_context |

## Custom events (Tier 2/3)

A hook can emit its own event name to coordinate with another hook in the same task. Naming convention:

```
iter<N>_<slug>_<event>        e.g. iter12_length_recovery_recovered
on_<thing>                    cross-iter failure-mode name
```

Declare `emits=("iter<N>_<slug>_<event>", ...)` on the publisher; subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow`/`inject_context` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent/component_runtime/policy.py`. Grep `agent/components/*.py` for `emits=` to find taken names.

## What this skill does NOT do

- Run benchmarks (the outer loop runs `run_benchmark.py` on train-30).
- Register new tools (capability expansion — a separate workflow).
- Modify `agent/component_runtime/`, `agent/tools/`, prior `agent/components/*.py` (except via `replace`), or any `agent/v*/`.
- Build a new agent directory.
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as a `mechanism_layer` override (`rewrite` / `block`). Route to `induced_rule` + `inject_context` instead.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
