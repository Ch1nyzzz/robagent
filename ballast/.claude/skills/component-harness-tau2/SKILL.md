---
name: component-harness-tau2
description: Propose ONE hook (a single Python file declaring `COMPONENT: Component`) that stabilizes the tau2-bench main agent on a recurring failure mode, plus a workflow patch (add / replace / disable) against `ballast/workflows/tau2_main.yaml`. The main agent runs the upstream tau2 LLMAgent multi-turn FC loop against a domain policy + tool catalog; hooks subscribe to lifecycle events via `listens=` and react via a Decision (allow / block / rewrite_tool_args / inject_context / defer). Admission gated by a class×event×decision matrix and a Trust block (evidence_anchor + out_of_evidence_probe).
---

# component-harness-tau2

The tau2-bench main agent is the protagonist. It runs upstream's `LLMAgent` multi-turn function-calling loop over the domain's full tool catalog + policy doc with the locked SUT model. **Your job is to stabilize it**, not replace it. You propose ONE hook — a single Python file under `agent_tau2/components/<name>.py` exporting `COMPONENT: Component` — plus a workflow patch (`add` / `replace` / `disable`). **You do NOT run the simulator.** You read prior simulations, write one hook, write `pending_eval.json`, exit. The outer loop scores it on train-30 and admits or rejects.

## First principles

`RESULTS.md §4` validated these. The GAIA +50% relative test gain came from hooks honoring all four; the tau2-banking regression (test 7/67 vs train 19/30) came from stacks that violated principle (3).

0. **Main agent is the protagonist.** Hooks stabilize its tool args / outputs / retries; they don't replace its policy reasoning. Before writing a hook, ask: "would the main agent still be the same agent without this — just less prone to failing on X?" If "no, it'd be doing something fundamentally different" (e.g. you're encoding the workflow decision tree in Python), the hook is too heavy.
1. **Capability vs Stabilization — keep them separate.** If the main agent is failing because it can't *reach* a domain doc / KB entry / external API, that's a capability gap — file it as a tool / sub-agent registration, **not** a hook that injects the content. Hooks are for stabilizing what the agent already can do: strip optional args the tool rejects, retry on observed tool errors, normalize a turn that came back empty.
2. **The LLM is the last resort within stabilization.** When you do write a hook, find one place the LLM is repeating deterministic work — arg shape validation, response retry on observed failure, output normalization — and move it into Python.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on a system field, the tool's declared JSON Schema, a protocol invariant, an LLM API field (`finish_reason`), or a general algorithm. A hook INDUCED from N failed sims is memorisation. `RESULTS.md §4.5`: 10 layers at ε=5% compound to ~40% test false-positive.
4. **Honest over fabricated.** When the agent cannot satisfy a request within policy, it says so and follows the escalation path — it does not invent an action or a confirmation.

## Toolbox (pick what fits — none of these are exclusive)

| When you'd reach for it | What you change |
|---|---|
| Main agent can't reach something it needs (new MCP, new KB, new tool) | Register a new tool / sub-agent on the baseline (Capability expansion — separate workflow, **not this skill**) |
| Main agent reaches it but mis-shapes the call | Write a `pre_tool_use` `mechanism_layer` (this skill) |
| Tool returns error and the LLM loops on identical args | Write an `on_tool_error` `reactive_guard` (this skill) |
| Assistant turn comes back empty / length-truncated | Write a `reactive_guard` on `on_length_truncation` / `on_empty_response` (this skill) |
| Static framework fact every session needs ("audit is discoverable; tools log to that table") | Write a `mechanism_layer` on `session_start` (this skill) |
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

### Lifecycle events the tau2 runtime emits

Per-task setup (fires once at agent construction):

| event | when it fires | typical use |
|---|---|---|
| `task_received` | start of `_collect_prompt_injection` | lifecycle anchor |
| `pre_context_build` | paired with the PRE_CONTEXT_BUILD phase | dynamic per-task framework injection |
| `session_start` | paired with SESSION_START phase | static framework-invariant injection |
| `pre_agent_construct` | last hook before instructions sealed | inference-hint injection |
| `user_prompt_submit` | incoming `UserMessage`, before the LLM sees it | reactive note / advisory injection |

Per LLM turn (`generate_next_message`):

| event | when it fires | typical use |
|---|---|---|
| `pre_llm_request` | **per turn**; just before each SUT call | last-pass per-turn injection |
| `post_llm_response` / `post_llm_response_raw` | **per turn**; after the assistant turn | sub-LLM verifier; clear / rewrite tool_calls |
| `on_length_truncation` | **per turn**; synthesised when `finish_reason=="length"` | sub-LLM recovery |
| `on_empty_response` | **per turn**; synthesised when content + tool_calls both empty | reactive retry |
| `on_no_tool_call_emitted` | **per turn**; synthesised when no tool_calls in assistant turn | retry-with-tool reminder |

Per ToolCall (each call in `assistant_message.tool_calls`):

| event | when it fires | typical use |
|---|---|---|
| `pre_tool_arg_validation` | narrow schema-check phase before `pre_tool_use` | rewrite args via REWRITE_TOOL_ARGS |
| `pre_tool_use` | per ToolCall, before it leaves the agent | wrap-tool / arg sanitization / BLOCK to drop one call |

Per ToolMessage (each tool result):

| event | when it fires | typical use |
|---|---|---|
| `post_tool_result_raw` | per ToolMessage in `_fire_post_tool_use` | observe |
| `on_tool_error` | **synthesised** when `tm.error` is set | retry hint via `inject_context` |
| `post_tool_use` | per ToolMessage, main post-tool decision | reactive guard on observed `tool.failed` |

Exit phase: `on_explicit_terminate`, `stop`, `session_end` are declared in policy but not yet dispatched (reserved). A component declaring `listens="stop"` loads but never fires today.

To scope a per-tool hook to one tool name, use `matcher_for_tool("close_bank_account")` from `agent_tau2.component_runtime.types` instead of writing a free-form matcher.

### Component classes

| class | what the matcher tests | risk |
|---|---|---|
| `mechanism_layer` | system field / tool-declared schema / protocol invariant / general algorithm | LOW |
| `reactive_guard` | observed failure event (`tool.failed`, empty turn, `finish_reason=length`) | LOW |
| `induced_rule` | reading of policy text — IF/THEN compiled from N=3-5 evidence sims | HIGH (advisory `inject_context` only) |
| `predictive_heuristic` | raw prompt text via regex / keywords | REJECTED at load time |

### Decision

| decision | semantics |
|---|---|
| `allow` | no-op |
| `block` | drop the tool_call (`pre_tool_use` / `pre_tool_arg_validation`), clear tool_calls (`post_llm_response`), or terminate (other events) |
| `rewrite_tool_args` | replace the in-flight ToolCall arguments (payload = `dict`) |
| `defer` | **v1: falls back to `allow` and traced.** A real replay queue is v2.5+ |
| `inject_context` | setup events → `ctx.proposed_system_prompt` (visible to next subscriber) + outer `tier1_prompt_inject`; post-LLM / post-tool events → next-turn SystemMessage |

Event → effect of decisions on tau2 objects (handled by `_apply_tau2_decision`):

| event family | INJECT_CONTEXT lands in | REWRITE_TOOL_ARGS effect | BLOCK effect |
|---|---|---|---|
| setup (`task_received` / `pre_context_build` / `session_start` / `pre_agent_construct` / `user_prompt_submit` / `pre_llm_request`) | `ctx.proposed_system_prompt` + outer `tier1_prompt_inject` | — | terminate task |
| `pre_tool_use` / `pre_tool_arg_validation` | (next-turn note) | rewrite the in-flight ToolCall args | drop this ToolCall |
| `post_llm_response[_raw]` | next-turn SystemMessage | rewrite first ToolCall args | clear all tool_calls |
| `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | next-turn SystemMessage | — | terminate task |

### Trust

Required: `evidence_anchor` / `blast_radius` (local|workflow|global) / `rollback_when`. `out_of_evidence_probe` is required for `induced_rule`. Optional: `fallback`.

- `evidence_anchor`: name a stable structure OUTSIDE your evidence — a system field, a tool's schema field, a protocol invariant, or a general algorithm. If your anchor is "I observed sim_017/023/041 all do X", you're anchored INSIDE evidence — pick a different class or don't write the hook.
- `out_of_evidence_probe` (induced_rule only): name one concrete case NOT in evidence sims where the matcher fires and what handler returns on it. If you can't, the rule overfits by construction.

### Handler helpers on `ctx`

| helper | what it does |
|---|---|
| `ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` | sub-LLM call via the locked SUT model |
| `ctx.shared` | per-task dict, free to read/write |
| `ctx.state[component_name]` | per-task per-component scratchpad |
| `ctx.emit("custom_event_name", **fields)` | fire custom event; re-enters dispatcher (depth cap 10) |
| `ctx.emit_upstream(key, value)` | shorthand for `ctx.upstream[key] = value` |

Per-event convenience fields: `ctx.tool_call` / `ctx.tool_name` / `ctx.tool_args` (on per-tool events), `ctx.incoming_message` (ToolMessage on post-tool events, UserMessage on user_prompt_submit), `ctx.assistant_message` (on post_llm_response), `ctx.history`, `ctx.domain_policy`, `ctx.tool_names`. NEVER read `ctx.task_id`.

## The class × event × decision matrix

Load-time gate at `agent_tau2/component_runtime/policy.py::ALLOWED`. A `(cls, listens)` not in the matrix → `ComponentPolicyError` at load. A handler returning a non-admitted Decision → raises at fire time.

| class \ event family | setup (`task_received` / `pre_context_build` / `session_start` / `pre_agent_construct` / `user_prompt_submit` / `pre_llm_request`) | `pre_tool_use` / `pre_tool_arg_validation` | `post_llm_response[_raw]` | `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | `on_length_truncation` / `on_empty_response` / `on_no_tool_call_emitted` | `on_explicit_terminate` / `stop` / `session_end` |
|---|---|---|---|---|---|---|
| `mechanism_layer` | inject_context | rewrite_tool_args, defer, block (pre_tool_use); rewrite_tool_args, block (pre_tool_arg_validation) | rewrite_tool_args, block, inject_context | inject_context | block, inject_context | block (terminate); allow (session_end) |
| `reactive_guard` | inject_context (`user_prompt_submit` only) | block, rewrite_tool_args | rewrite_tool_args, block, inject_context | inject_context | block, inject_context | block (terminate) |
| `induced_rule` | **inject_context (advisory)** on `pre_context_build` / `user_prompt_submit` only | — | — | — | — | — |
| `predictive_heuristic` | rejected | rejected | rejected | rejected | rejected | rejected |

`induced_rule` is admitted strictly advisory: a policy reading as `inject_context` is safer than the LLM rediscovering the passage cold (worst case = redundant prompt). `predictive_heuristic` is rejected: even advisory injection on a prompt-text regex conditions interpretation on raw surface text — structurally unsafe.

## The workflow file

```yaml
# ballast/workflows/tau2_main.yaml
nodes:
  - close_account_strip_optional_reason
  - discoverable_audit_channel
  - cc_account_workflow_doc_index
disabled: []
```

Plus a JSON snapshot at `ballast/logs_tau2_components/frontier_workflow.json` (written ONLY on acceptance).

### Patch ops

| op | meaning |
|---|---|
| `add` | append a new node; `agent_tau2/components/<id>.py` must be newly written |
| `replace` | keep the existing id; overwrite the file (same `COMPONENT.name`) |
| `disable` | move id into `disabled:`; file remains for the durability audit |

For `replace`, the **first shell action MUST be** `cp agent_tau2/components/<existing>.py agent_tau2/components/<existing>.py.bak_iter<N>` — the outer loop relies on the `.bak` to roll back on reject.

Sugar derivable from `add`: `wrap_tool(tool_name)` = `add` with `listens="pre_tool_use"` and `matcher_for_tool(tool_name)`. Ordering within an event bucket is controlled by `priority` (v1 dispatch is event-driven, not edge-driven).

## Hard rules

- Exactly ONE patch per invocation. One of `add` / `replace` / `disable`.
- **You do NOT run the simulator.** No `tau2_runner.py`, no `tau2 run`. The outer loop scores.
- **No task-specific code.** No customer names, account / document ids, per-task branching, no encoded gold answers or gold action sets.
- General documented policy may enter as **advisory context** via `induced_rule` on `pre_context_build` / `user_prompt_submit` (`inject_context` only).
- **The target inference model is LOCKED via the tau2 LLM config.** Hooks may NOT spin up a different model. `ctx.chat()` IS permitted for sub-LLM verifier patterns — it routes through `agent.llm.chat` (locked SUT model name) with mutable inference params.
- **DEFER is v1-degraded.** It falls back to `allow` and traces; do not design a hook whose correctness depends on real deferral until v2.5.
- **Capability gaps are not for hooks.** If the main agent needs a new tool / KB / external API, file a separate request — do not invent a hook that injects content the agent should reach itself.
- For `replace`, the **first shell action** MUST be `cp agent_tau2/components/<existing>.py agent_tau2/components/<existing>.py.bak_iter<N>`.
- READ-ONLY: `tau2-bench-src/`, `tau2_runner.py`, `ballast/*`, `agent_tau2/v0/`, `agent_tau2/component_runtime/`, all earlier `agent_tau2/components/*.py` (modify only via `replace`), all `ballast/logs_tau2_*/` directories EXCEPT `logs_tau2_components/`.
- You may NOT create a new agent directory. Your only file writes are: ONE component file under `agent_tau2/components/<name>.py` (+ its `.bak_iter<N>` for `replace`) and the `pending_eval.json` manifest.

## Component file template

```python
# agent_tau2/components/<name>.py
from __future__ import annotations

from agent_tau2.component_runtime.types import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.tool_name / ctx.tool_args (per-tool events) /
    # ctx.assistant_message (post_llm_response) / ctx.incoming_message
    # (post_tool_use; user_prompt_submit) / ctx.history /
    # ctx.domain_policy / ctx.tool_names. ctx.event names the firing event.
    # NEVER read ctx.task_id (hidden at fire time).
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite_tool_args({...})   # or inject_context / block / allow / defer


COMPONENT = Component(
    name="<stable_component_id>",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_use",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="<system field / tool schema / protocol invariant / general algorithm>",
        blast_radius="local",
        rollback_when="<observable rollback signal>",
        out_of_evidence_probe="<required for induced_rule; concrete OOE case + handler return>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

Reference examples already in the frontier:
- `agent_tau2/components/close_account_strip_optional_reason.py` — `mechanism_layer` / `pre_tool_use` / `rewrite_tool_args`
- `agent_tau2/components/discoverable_audit_channel.py` — `mechanism_layer` / `session_start` / `inject_context`

## Workflow

### 1. Understand the domain (before any failure analysis)

Read the domain's policy document(s), the full tool catalog, and protocol invariants. Write a workflow summary in your builder log: the end-to-end happy path, mandatory orderings, prerequisite reads, KB structure, session-state requirements. Form hypotheses against this reference frame.

### 2. Read state and failed simulations

```
ballast/workflows/tau2_main.yaml                                active workflow
ballast/logs_tau2_components/frontier_workflow.json             frontier snapshot
ballast/logs_tau2_components/frontier_val.json                  per-task best
ballast/logs_tau2_components/evolution_summary.jsonl            one row per iter (incl. rejected)
ballast/tau2_train_task_ids.txt                                 30 tasks — your pool
agent_tau2/components/                                                hooks on disk
.component-state/iter<K>/fired.jsonl                                 which hooks fired in iter K
```

Per-iter simulation dumps + summaries (preserved across iters):

```
tau2-bench-src/data/simulations/tau2-runs/meta/v0__<domain>.json/results.json       v0 baseline / iter 0
tau2-bench-src/data/simulations/tau2-runs/meta/iter<K>/component_runtime__<domain>.json/results.json   iter K's candidate run
traces/iter<K>__tau2_component_runtime__summary.jsonl                               iter K summary jsonl
```

Each `results.json` carries `simulations[].messages` / `simulations[].reward_info` for every task. Pick 4-6 train tasks the frontier still fails (reward 0) and open the appropriate iter's `results.json`. Cross-reference `.component-state/iter<K>/fired.jsonl`. Skip tasks whose failure is "no tool exists for this kind of action" — that's a capability gap, not a stabilization problem.

### 3. Form ONE hypothesis

```
HYPOTHESIS:            <falsifiable claim about train-30 reward>
MECHANISM:             <failure mode> seen in N≥3 task sims [tid1, tid2, ...]
WHY STABILIZATION:     <why this is a stabilization gap, not a missing capability>
STABLE STRUCTURE:      <evidence_anchor — system field / tool schema / protocol invariant /
                        general algorithm; must live OUTSIDE the N evidence sims>
OUT_OF_EVIDENCE PROBE: <required for induced_rule; concrete OOE case + handler return>
PATCH_OP:              <add | replace | disable>
COMPONENT:             listens=<...>, cls=<...>
EXPECTED_DELTA:        train-30 reward <current> → <expected>
```

If you can't name a STABLE STRUCTURE outside evidence, or your hook is essentially "give the agent content / decisions it should reach itself" → re-examine. The latter is a tool / policy request, not a hook.

### 4. Classify class + choose event

| matcher tests… | class | suggested events |
|---|---|---|
| system field / tool schema field / protocol invariant | `mechanism_layer` | `pre_tool_use` / `pre_tool_arg_validation` |
| observed failure event in `ctx.incoming_message` (ToolMessage.error) | `reactive_guard` | `post_tool_use` / `on_tool_error` |
| LLM API field (`finish_reason=length`, empty content) | `reactive_guard` | `on_length_truncation` / `on_empty_response` |
| consistency-check across assistant turn (text-vs-tool_call mismatch) | `mechanism_layer` | `post_llm_response` (sub-LLM verifier) |
| framework constant ("audit is discoverable") | `mechanism_layer` | `session_start` (inject_context) |
| policy/instruction reading (advisory) | `induced_rule` | `pre_context_build` / `user_prompt_submit` (inject_context only) |
| raw prompt text via regex/keywords | none — redesign or don't write | — |

### 5. Implement + validate

Create exactly one file at `agent_tau2/components/<name>.py`. Validate:

```bash
python -c "
from agent_tau2.component_runtime.registry import load_components_from_dir
comps = load_components_from_dir(only=['<COMPONENT.name>'])
assert any(c.name == '<COMPONENT.name>' for c in comps)
print('loads + passes policy:', True)
"
```

For `replace`, FIRST run the `cp ... .bak_iter<N>` command. This is the ONLY shell command you run.

### 6. Choose patch op

| situation | op |
|---|---|
| no existing node addresses this mechanism | `add` |
| an existing node addresses this mechanism but has a known bug | `replace` (reuse `COMPONENT.name`; .bak first) |
| an existing node is provably dead weight or actively harmful | `disable` (no new file; reference existing id) |

### 7. Write `pending_eval.json`

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected reward change on train-30>",
    "component": {
      "id": "<stable_component_id>",
      "cls": "mechanism_layer | reactive_guard | induced_rule",
      "listens": "<event_name>",
      "file": "agent_tau2/components/<name>.py",
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
      "name": "<component name; existing id for disable>",
      "file": "agent_tau2/components/<name>.py"
    }
  }
}
```

For `disable`, the `component` block can be omitted; only `workflow_patch.name` matters.

### 8. Session log + exit

Write a concise log to `ballast/logs_tau2_components/builder_sessions/iter<N>/log.md`. Final line of your reply:

```
CANDIDATE: candidate_iter<N>_<slug>
```

## Common stabilization patterns

| pattern | class | listens | decision |
|---|---|---|---|
| strip an optional arg the tool rejects | mechanism_layer | `pre_tool_use` | rewrite_tool_args |
| wrap one tool's args (matcher = `matcher_for_tool(...)`) | mechanism_layer | `pre_tool_use` | rewrite_tool_args |
| schema-validate args narrowly (drop unknown kwargs) | mechanism_layer | `pre_tool_arg_validation` | rewrite_tool_args |
| block a malformed tool call | reactive_guard | `pre_tool_use` | block (drops this ToolCall only) |
| retry hint on `tool.failed` | reactive_guard | `on_tool_error` | inject_context (next-turn note) |
| consistency check (assistant text vs tool_call) | mechanism_layer | `post_llm_response` | block / rewrite_tool_args (sub-LLM verifier) |
| length-recovery via sub-LLM with bigger budget | reactive_guard | `on_length_truncation` | inject_context or rewrite via `ctx.chat(max_tokens=32768)` |
| inject framework constant ("audit is discoverable") | mechanism_layer | `session_start` | inject_context |
| advisory policy note ("closure flow has prerequisites; consult doc_021") | induced_rule | `pre_context_build` | inject_context |
| reactive note on incoming user message | reactive_guard | `user_prompt_submit` | inject_context |
| two-hook coordination | mechanism_layer | A emits `iter<N>_<slug>_X` → B `listens="iter<N>_<slug>_X"` (reads `ctx.upstream`) | allow / inject_context |

## Custom events (Tier 2/3)

A hook can emit its own event name to coordinate with another hook in the same task. Naming convention:

```
iter<N>_<slug>_<event>        e.g. iter12_audit_visibility_resolved
on_<thing>                    cross-iter failure-mode name
```

Declare `emits=("iter<N>_<slug>_<event>", ...)` on the publisher; subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` / `inject_context` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent_tau2/component_runtime/policy.py`. Grep `agent_tau2/components/*.py` for `emits=` to find taken names.

## What this skill does NOT do

- Run the tau2 simulator (the outer loop runs `tau2_runner.py` on train-30).
- Register new tools / MCPs / sub-agents (capability expansion — a separate workflow).
- Modify `tau2-bench-src/`, the eval, the model, `tau2_runner.py`, `ballast/`, `agent_tau2/v0/`, `agent_tau2/component_runtime/`, prior component files (except via `replace` + `.bak` protocol).
- Build a full `LLMAgent` subclass (that pattern was retired in favour of components).
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as a `mechanism_layer` override (`rewrite_tool_args` / `block`). Route to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
- Design around `defer` semantics — v1 falls back to `allow`.
