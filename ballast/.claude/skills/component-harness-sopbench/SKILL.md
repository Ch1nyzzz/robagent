---
name: component-harness-sopbench
description: Propose ONE hook (a single Python file declaring `COMPONENT: Component`) that stabilizes the SOP-Bench main agent on a recurring failure mode for ONE domain, plus a workflow patch (add / replace / disable) against `ballast/workflows/sopbench_<domain>.yaml`. The main agent runs a multi-turn FC loop over the domain's SOP doc + tool catalog, then emits an XML report; hooks subscribe to lifecycle events via `listens=` and react via a Decision (allow / block / rewrite / inject_context). Admission gated by a class×event×decision matrix and a Trust block (evidence_anchor + out_of_evidence_probe).
---

# component-harness-sopbench

The SOP-Bench main agent is the protagonist. For each domain (`dangerous_goods`, `traffic_spoofing_detection`, …) it runs a multi-turn function-calling loop over the SOP doc + per-domain ToolManager with the locked SUT model and emits an XML-tagged report. **Your job is to stabilize it**, not replace it. You propose ONE hook — a single Python file under `agent/components_sopbench_<domain>/<name>.py` exporting `COMPONENT: Component` — plus a workflow patch (`add` / `replace` / `disable`). **You do NOT run benchmarks.** You read prior baselines + traces, write one hook, write `pending_eval.json`, exit. The outer loop applies the patch, scores it on the domain's 30-row train subset, and admits or rejects.

## What SOP-Bench tasks look like

A task = one row from `third_party/SOP-Bench/src/amazon_sop_bench/benchmarks/data/<domain>/test_set_with_outputs.csv`. The agent receives:
  * `sop` (str) — the natural-language SOP document (visible via `ctx.sop_text`)
  * `task` (dict) — structured inputs (per `metadata.json::input_columns`; visible via `ctx.task_input`)
  * `tools` (ToolManager) — function-callable utilities the SOP says to call (specs in `ctx.tool_specs`)

The agent runs a multi-turn FC loop, then emits XML like `<hazard_class>Hazard Class B</hazard_class>`. Common failure modes:
  * **format / case mismatch** — model writes `hazard class b`; grader expects `Hazard Class B`.
  * **incorrect reasoning** — wrong branch of the SOP decision tree.
  * **tool-arg hallucination** — passes a parameter the tool spec doesn't accept.
  * **missing tool call** — answers without invoking the SOP-prescribed validation step.
  * **format guard** — XML report with the right value buried in extra tags.

## First principles

0. **Main agent is the protagonist.** Hooks stabilize its arg shapes / output format / retries; they don't replace its SOP-following reasoning. Before writing a hook, ask: "would the main agent still be the same agent without this — just less prone to failing on X?" If "no, it'd be doing fundamentally different work" (e.g. you're encoding the SOP decision tree in Python), the hook is too heavy — that's a capability replacement, not stabilization.
1. **Capability vs Stabilization — keep them separate.** If a domain is failing because the SOP needs a tool the baseline ToolManager doesn't expose, that's a capability gap — register a tool. Hooks are for stabilizing what the agent already can do: drop unknown kwargs the tool rejects, title-case the XML inner text, retry on observed errors.
2. **The LLM is the last resort within stabilization.** When you do write a hook, find one place the LLM is doing work deterministic code could do — output normalisation, tool-arg validation, format extraction, retry-on-failure — and move it into Python.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on a tool's declared JSON Schema, an XML grammar invariant, the SOP's quoted output format, an LLM API field. IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Per-domain specialization.** Components live in per-domain dirs (`agent/components_sopbench_<domain>/`) and are namespaced by domain. Your work is for ONE domain only; do not optimize for cross-domain transfer.

## Toolbox (pick what fits — none of these are exclusive)

| When you'd reach for it | What you change |
|---|---|
| Main agent can't reach a tool the SOP requires | Register a tool in the domain's ToolManager (Capability expansion — separate workflow, **not this skill**) |
| Main agent calls a tool with a shape it rejects | Write a `pre_tool_arg_validation` / `pre_tool_use` `mechanism_layer` (this skill) |
| Tool errors and the LLM loops on identical args | Write an `on_tool_error` `reactive_guard` (this skill) |
| Final XML report has the right value but wrong case / extra tags | Write a `pre_final_emit` `mechanism_layer` (this skill) |
| Static framework reminder ("output in `<answer>X</answer>` format") | Write a `mechanism_layer` on `session_start` (this skill) |
| Multiple hooks need to coordinate | Custom event: A `ctx.emit("iter<N>_<slug>_X")`, B `listens="iter<N>_<slug>_X"` |

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

### Lifecycle events the SOP-Bench runtime emits

Per-task setup:

| event | when it fires | typical use |
|---|---|---|
| `task_received` | right after `ComponentContext` built | static framework injection |
| `session_start` | once per task | static system-prompt injection |
| `pre_prompt_build` | per task; default system+user prompts built | rewrite user prompt; inject SOP step checklist |
| `pre_context_build` | alias of `pre_prompt_build` (cross-sibling vocabulary) | same as above |
| `pre_agent_construct` | last hook before messages list sealed | inference-hint injection |

Per LLM turn (fires multiple times per task):

| event | when it fires | typical use |
|---|---|---|
| `pre_llm_turn` | per turn, before `chat()` | rewrite messages list |
| `pre_llm_request` | per turn, just before SUT call (alias for `pre_llm_turn`) | sub-LLM verifier prep |
| `post_llm_response` / `post_llm_response_raw` | per turn, after chat | rewrite assistant content; trigger retry |
| `on_length_truncation` | **synthesised** when `finish_reason=="length"` | sub-LLM recovery with bigger budget |
| `on_empty_response` | **synthesised** when raw_response empty | reactive retry |
| `on_no_tool_call_emitted` | **synthesised** when no tool_calls | retry-with-tool reminder |

Per tool call (once per ToolCall within a turn):

| event | when it fires | typical use |
|---|---|---|
| `pre_tool_arg_validation` | narrow schema-check phase before `pre_tool_use` | rewrite args (drop unknown kwargs) / block |
| `pre_tool_use` | main pre-tool decision | validate / rewrite args; block invalid calls |
| `post_tool_use` | after invocation | reformat result; insert validation flag |
| `post_tool_result_raw` | raw result anchor | observe / inject |
| `on_tool_error` | **synthesised** when `current_tool_success==False` | retry-hint via `inject_context` |

Per-task exit:

| event | when it fires | typical use |
|---|---|---|
| `on_explicit_terminate` | **synthesised** when max_iterations exhaust without final answer | exit-gate; may BLOCK to refuse termination |
| `pre_final_emit` | after the loop, before returning final XML | normalise XML casing; ensure required tag present |
| `session_end` | bookkeeping | — |

Helper: `matcher_for_tool("tool_name")` from `agent.component_runtime_sopbench.types` scopes per-tool matchers cleanly.

### Component classes

| class | what the matcher tests | risk |
|---|---|---|
| `mechanism_layer` | system field / tool-declared JSON Schema / LLM API field / XML grammar / general algorithm | LOW |
| `reactive_guard` | observed failure event (tool error, malformed args, empty content, missing XML tag) | LOW |
| `induced_rule` | reading of SOP text — IF/THEN compiled from N evidence rows | HIGH (advisory `inject_context` only) |
| `predictive_heuristic` | raw prompt / response text via regex / keywords | REJECTED at load time |

### Decision

| decision | semantics |
|---|---|
| `allow` | no-op |
| `block` | terminate task as blocked (final answer empty) — EXCEPT at `pre_tool_use` / `pre_tool_arg_validation` where it skips just this tool call |
| `rewrite` | replace the live payload at this event (see Event → payload below) |
| `inject_context` | setup events → append to `ctx.system_prompt`; per-turn / per-tool events → queue for next turn |

Event → REWRITE payload:

| event | payload type | replaces |
|---|---|---|
| `pre_prompt_build` / `pre_context_build` | `str` | `ctx.user_prompt` |
| `pre_llm_turn` | `list[dict]` | `ctx.messages` |
| `post_llm_response[_raw]` / `on_length_truncation` / `on_empty_response` / `on_no_tool_call_emitted` | `str` | `ctx.raw_response` |
| `pre_tool_use` / `pre_tool_arg_validation` | `dict` | `ctx.current_tool_args` |
| `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | `str` | `ctx.current_tool_result_str` |
| `pre_final_emit` | `str` or `None` | `ctx.final_output` (None marks blocked) |

### Trust

Required: `evidence_anchor` / `blast_radius` (local|workflow|global) / `rollback_when`. `out_of_evidence_probe` required for `induced_rule`. Optional: `fallback`.

- `evidence_anchor`: name a stable structure OUTSIDE evidence — a tool's JSON Schema field, the SOP's quoted Output section grammar, an OpenAI/DeepSeek API field, an XML grammar invariant. If your anchor is "I observed row_017/023/041 all do X", you're anchored INSIDE evidence — pick a different class or don't write the hook.
- `out_of_evidence_probe` (induced_rule only): name one concrete case NOT in evidence rows where the matcher fires and what handler returns on it.

### Handler helpers on `ctx`

| helper | what it does |
|---|---|
| `ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` | sub-LLM call via the locked SUT model |
| `ctx.shared` | per-task dict, free to read/write |
| `ctx.state[component_name]` | per-task per-component scratchpad |
| `ctx.emit("custom_event_name", **fields)` | fire custom event; re-enters dispatcher (depth cap 10) |
| `ctx.emit_upstream(key, value)` | shorthand for `ctx.upstream[key] = value` |

Per-event fields: `ctx.benchmark` (domain slug), `ctx.sop_text` (the SOP doc), `ctx.task_input` (structured task dict), `ctx.tool_specs` (bedrock-format toolspecs), `ctx.system_prompt`, `ctx.user_prompt`, `ctx.messages`, `ctx.raw_response`, `ctx.tool_calls`, `ctx.turn_index`, `ctx.finish_reason`, `ctx.current_tool_name / _args / _result / _result_str / _call_id / _success / _error`, `ctx.final_output`. NEVER read `ctx.task_id`. NEVER compare `ctx.task_input` fields against any encoded target — that's memorisation of the test set.

## The class × event × decision matrix

Load-time gate at `agent/component_runtime_sopbench/policy.py::ALLOWED`. A `(cls, listens)` not in the matrix → `ComponentPolicyError` at load.

Setup events (`task_received`, `session_start`, `pre_prompt_build`, `pre_context_build`, `pre_agent_construct`):

| class | admissible decisions |
|---|---|
| `mechanism_layer` | inject_context (+ rewrite / block on `pre_prompt_build` / `pre_context_build`) |
| `induced_rule` | inject_context (`pre_prompt_build` / `pre_context_build` only — ADVISORY) |
| `reactive_guard` | — (does not fire at setup) |

Per-turn / per-tool events:

| class | admissible decisions |
|---|---|
| `mechanism_layer` | rewrite / block / inject_context (where the cell admits each — see policy.py) |
| `reactive_guard` | rewrite / block / inject_context (where the cell admits each) |
| `induced_rule` | — (does not fire on these events) |

Exit events (`on_explicit_terminate`, `pre_final_emit`, `session_end`):

| class | admissible decisions |
|---|---|
| `mechanism_layer` | rewrite + block (`pre_final_emit`); block (`on_explicit_terminate`); allow (`session_end`) |
| `reactive_guard` | same |

`predictive_heuristic` is rejected at every event.

## The workflow file

Per-domain frontier YAML: `ballast/workflows/sopbench_<domain>.yaml`.

```yaml
nodes:
  - sopbench_dangerous_goods_final_xml_recovery
disabled: []
```

Plus a JSON snapshot `ballast/logs_components_sopbench_<domain>/frontier_workflow.json`.

### Patch ops

| op | meaning |
|---|---|
| `add` | append a new node; `agent/components_sopbench_<domain>/<id>.py` must be newly written |
| `replace` | keep the existing id; overwrite the file (same `COMPONENT.name`) |
| `disable` | add the id to `disabled:`; file remains for durability audit |

For `replace`, the **first shell action MUST be** `cp agent/components_sopbench_<domain>/<existing>.py agent/components_sopbench_<domain>/<existing>.py.bak_iter<N>`.

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_sopbench_baseline.py`, no `evaluate()`. The outer loop scores.
- **No task-specific code.** No domain-specific entity literals in matchers (no chemical IDs, partner IDs, PO numbers). No encoded gold answers.
- **The target inference model is LOCKED.** `agent.llm.chat()` rejects any `model=` override. `ctx.chat()` does not accept a `model` kwarg.
- **Capability gaps are not for hooks.** If the SOP needs a tool the baseline ToolManager doesn't expose, register the tool — do not invent a hook that injects the would-be tool result.
- **Per-domain dir is mandatory.** Files MUST live under `agent/components_sopbench_<domain>/`; the per-domain registry refuses to load components from another domain's dir.
- For `replace`, the **first shell action** MUST be the `cp ... .bak_iter<N>` command above.
- READ-ONLY: `third_party/SOP-Bench/`, `amazon_sop_bench/`, `bench/`, `agent/sopbench_agent.py`, `agent/component_runtime_sopbench/`, `agent/llm.py`, `agent/events.py`, all other domains' `agent/components_sopbench_*/`, `agent/components/` (GAIA's), `ballast/scripts/run_sopbench_baseline.py`, `ballast/evolve_sopbench.py`.
- Component file naming: `agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py`. The `COMPONENT.name` SHOULD also include the domain slug for grep-ability.

## Component file template

```python
# agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime_sopbench import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_name /
    # ctx.current_tool_args / ctx.current_tool_result_str / ctx.finish_reason /
    # ctx.sop_text / ctx.tool_specs / ctx.shared.
    # ctx.event names the firing event for branchable handlers.
    # NEVER read ctx.task_id.
    # NEVER compare ctx.task_input fields against an encoded target — that's
    # memorisation of the test set.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)        # or inject_context / block / allow


COMPONENT = Component(
    name="sopbench_<domain>_<slug>",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="pre_final_emit",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="<tool JSON Schema / XML grammar / SOP Output-section / API field>",
        blast_radius="local",
        rollback_when="<observable rollback signal>",
        out_of_evidence_probe="<required for induced_rule>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

## `pending_eval.json` schema

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "<one-sentence falsifiable claim of the failure mode>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<rid1>", "<rid2>", "<rid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected acc change on train-30>",
    "component": {
      "id": "<COMPONENT.name>",
      "cls": "mechanism_layer | reactive_guard | induced_rule",
      "listens": "<event_name>",
      "file": "agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py",
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
      "name": "<COMPONENT.name; existing id for disable>",
      "file": "agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py"
    }
  }
}
```

For `disable`, omit `component` and `workflow_patch.file`; only `workflow_patch.name` matters.

## How to investigate

1. **Read `frontier_val.json`'s `per_task`** at `ballast/logs_components_sopbench_<domain>/frontier_val.json` to find tasks the frontier scores 0 on. Each entry has `agent`, `score`, `predicted`.
2. **Read v0 baseline `results.json`** at `ballast/logs_components_sopbench_<domain>/v0__train__results.json`. Each task result has `task_id`, `success`, `predicted_output`, `expected_output`, `tool_calls`, `error`, `reasoning_trace`.
3. **Look for ≥3 failures sharing the same mechanism.** Examples:
   - All 3 outputs are correctly-named XML tags but lowercase → `pre_final_emit` REWRITE that title-cases the inner text. Anchor: XML grammar + the SOP's quoted Output section format.
   - All 3 failures call `calculate_X_score` with an extra `assessmentFormId` param the tool rejects → `pre_tool_arg_validation` REWRITE that strips kwargs not in the tool's JSON Schema. Anchor: the tool's declared schema.
   - All 3 failures emit prose without any XML tag → `post_llm_response` REACTIVE_GUARD that on regex-miss `inject_context` a retry hint. Anchor: the XML tag's literal name from the SOP.
4. **Form ONE hypothesis** tied to a stable structure (see anchors above). State it in `trust.evidence_anchor`.
5. **Write ONE hook** with the smallest possible matcher/handler. Resist embedding domain entity literals — look them up from `ctx.task_input` / `ctx.sop_text` at fire time.
6. **Validate**:
   ```bash
   python -c "
   from agent.component_runtime_sopbench import load_components_from_dir
   from pathlib import Path
   comps = load_components_from_dir(
       Path('agent/components_sopbench_<domain>'),
       only=['<COMPONENT.name>'],
   )
   assert any(c.name == '<COMPONENT.name>' for c in comps)
   print('ok')
   "
   ```
7. **Write `pending_eval.json`** and print `CANDIDATE: <name>`.

## Common stabilization patterns

| pattern | class | listens | decision |
|---|---|---|---|
| inject SOP output-format hint | mechanism_layer | `session_start` | inject_context |
| inject SOP step checklist | mechanism_layer | `pre_prompt_build` | inject_context |
| advisory: "the SOP's section 3 applies when X" | induced_rule | `pre_prompt_build` | inject_context |
| schema-validate tool args (drop unknown kwargs) | mechanism_layer | `pre_tool_arg_validation` | rewrite |
| block tool call with missing required param | reactive_guard | `pre_tool_use` | block (skips this tool call only) |
| retry hint on tool error | reactive_guard | `on_tool_error` | inject_context |
| length-recovery via sub-LLM with bigger budget | reactive_guard | `on_length_truncation` | rewrite via `ctx.chat(max_tokens=32768)` |
| regex-miss on required XML tag → retry hint | reactive_guard | `post_llm_response` | inject_context |
| ensure required XML tag exists at final emit | reactive_guard | `pre_final_emit` | rewrite / block |
| title-case XML inner text | mechanism_layer | `pre_final_emit` | rewrite |
| refuse termination without artifact | reactive_guard | `on_explicit_terminate` | block |

## Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_tool_arg_schema_filter_dropped_unknown_kwarg
on_<thing>                    cross-iter failure-mode name
```

Declare `emits=("iter<N>_<slug>_<event>", ...)` on the publisher; subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent/component_runtime_sopbench/policy.py`. Grep `agent/components_sopbench_<domain>/*.py` for `emits=` to find taken names.

## What this skill does NOT do

- Run benchmarks (the outer loop does).
- Register new tools (capability expansion — a separate workflow).
- Modify `agent/sopbench_agent.py`, `agent/component_runtime_sopbench/`, `agent/llm.py`, other domains' component dirs, or any prior component file (except via `replace` + `.bak` protocol).
- Memorise train-set answers (encode `hazard_class = "C"` for product_ids starting with `P_13`).
- Compute the SOP output deterministically without the LLM (defeats the SUT measurement).
- Fire on every task (`matcher=None`, priority=0) — that's effectively a SOP rewrite, not a hook.
- Loop or propose multiple patches in one invocation.
- Encode a SOP interpretation as `mechanism_layer` override (`rewrite` / `block`). Route to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
