---
name: component-harness-tau2
description: Run ONE iteration of tau2-bench harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow. The component is typed by mount + class + state_scope, gated by a class×mount×decision permission matrix, and verified by a Trust block (evidence_anchor + out_of_evidence_probe). Successor to hook-harness-tau2; replaces the flat hook lifecycle with a typed workflow-graph runtime that supports retriever sub-pipelines, sub-LLM verifier returns, and explicit graph patches.
---

# component-harness-tau2

Run ONE iteration of agent evolution against tau2-bench by proposing ONE **workflow graph patch**. A patch is one of `add_node` / `replace_node` / `disable_node`, applied to the frontier workflow graph. The node added or replaced is one **component** — a single Python file in `agent_tau2/components/<name>.py` exporting `COMPONENT: Component`. **You do NOT run the simulator.** You understand the workflow domain, analyse prior simulations, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop composes `frontier_workflow + your patch`, scores it on the train-30 subset, and either admits the patch into the frontier or rejects it.

## Why workflow graph (vs. the hook system)

`hook-harness-tau2` modelled candidates as **hooks** attached to lifecycle events (`session_start`, `pre_tool_use`, `post_tool_use`). That handled point interventions well — strip an arg, inject a constant — but couldn't express three operations the train→test gap in `RESULTS.md` pushed us toward:

1. **Retriever sub-pipelines.** A KB lookup must (a) fire on a fresh question, (b) call a tool, (c) inject the result. Hooks have no notion of "wait for my own sub-step." Components declare `mount=PRE_CONTEXT_BUILD` and own a multi-step handler with `capabilities=(TOOL_CALL,)`.
2. **Sub-LLM verifier returns.** A self-check that fires after the LLM emits a turn, runs an isolated mini-LLM, and rewrites or rejects the tool calls — hooks could observe `post_tool_use` but couldn't intercept the LLM's *own* output. Components mount at `POST_LLM_RESPONSE`.
3. **Graph replace.** Hooks chained by event but never by *position*: you couldn't say "replace the node that currently does X with this new implementation, keeping its edges." The workflow graph names nodes and supports `replace_node` directly.

Components also carry **explicit `capabilities` and `state_scope`**, so misdeclared side-effects fail at load time rather than at first fire.

## First principles

Unchanged from `hook-harness-tau2`. `RESULTS.md §4` validated them: the GAIA win (+50% relative on test) came from components honoring all four; the tau2-banking regression (test 7/67 vs train 19/30) came from stacks that honored (1) and (2) but violated (3).

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — parsing, arithmetic, lookup, comparison, ranking, sequencing, state tracking — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split it: what is the *minimal* judgment the LLM must make, and what computation / lookup / validation / sequencing around it is fully deterministic? Build that deterministic part as a component.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact that lives OUTSIDE your training evidence — a system field, a tool's declared schema, a protocol invariant, a general algorithm. A component INDUCED from the N failed simulations you read is a memorised map of those N simulations. `RESULTS.md §4.5` does the math: 10 layers of ε=5% interpretation-layer code compound to ~40% test-time false-positive.
4. **Honest over fabricated.** When the agent cannot satisfy a request within policy, it says so and follows the escalation path — it does not invent an action or a confirmation.

## The component model

A component is a single immutable record:

```python
@dataclass(frozen=True)
class Component:
    name: str                              # stable id; reuse for replace_node
    cls: ComponentClass                    # mechanism_layer | reactive_guard | channel | induced_rule
    mount: Mount                           # where in the workflow it fires
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust                           # required verification block
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100                    # smaller fires first within a mount
```

### Mount enum

| mount                 | when it fires                                                          | typical use                                    |
|-----------------------|------------------------------------------------------------------------|------------------------------------------------|
| `pre_context_build`   | **NEW** — before `get_init_state` freezes `system_prompt`              | dynamic channel retrieval per session          |
| `session_start`       | once, before any LLM call (extends `system_prompt`)                    | static framework-invariant injection           |
| `user_prompt_submit`  | incoming `UserMessage`, before the LLM sees it                         | reactive notes / advisory injection (reserved) |
| `pre_tool_use`        | per `ToolCall` the LLM emitted, before it leaves the agent             | wrap-tool, arg sanitization, blocking          |
| `post_llm_response`   | **NEW** — after `AssistantMessage` emitted, before tool dispatch       | sub-LLM verifier, content observer             |
| `post_tool_use`       | each incoming `ToolMessage` from the env                               | reactive guard on observed `tool.failed`       |
| `stop`                | the agent attempts to end the turn                                     | escalation gate (reserved)                     |
| `session_end`         | once, after the final message                                          | bookkeeping (reserved)                         |

v1 actually dispatches `pre_context_build`, `session_start`, `pre_tool_use`, `post_llm_response`, `post_tool_use`. The other three are declared and load-time validated but not yet dispatched.

`wrap_tool` is NOT a Mount. To scope a component to one tool name, use `mount=PRE_TOOL_USE` and a matcher that tests `ctx.tool_name == "<name>"`. The helper `matcher_for_tool("foo")` in `component_runtime.types` builds the predicate.

### ComponentClass enum

| class                 | the matcher tests…                                                                                                          | risk                          |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`     | a system field, a tool-declared schema, a protocol invariant, a general algorithm — facts that hold off-evidence            | LOW                           |
| `reactive_guard`      | an observed failure event — `tool.failed`, malformed call, empty / looping turn                                              | LOW                           |
| `channel`             | task structure — a request needing content the agent cannot otherwise reach (file, URL, KB doc)                              | LOW                           |
| `induced_rule`        | a reading of policy text — IF/THEN compiled from N=3-5 train sim observations                                                | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic`| raw prompt text — regex / keyword guess about what the model is about to do                                                  | REJECTED at load time         |

### Decision

| decision              | semantics                                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------|
| `allow`               | no-op (the no-op is always permitted)                                                      |
| `block`               | drop the tool_call (PRE_TOOL_USE / POST_LLM_RESPONSE) or terminate the turn (STOP)         |
| `rewrite_tool_args`   | replace the tool_call's arguments with the payload dict                                    |
| `defer`               | postpone the call until predicate fires (v1: falls back to `allow` and traced)             |
| `inject_context`      | append text to system_prompt (prebuild / session_start) or next-turn system note (others)  |

### StateScope

| scope             | semantics                                                                              |
|-------------------|----------------------------------------------------------------------------------------|
| `none`            | pure function of `ctx`; no persistence                                                 |
| `session`         | per-component dict at `ctx.state[component_name]`; cleared at session end              |
| `cross_session`   | persisted under `.component-state/<run_tag>/`; reserved — not enforced in v1           |

### Trust

```python
@dataclass(frozen=True)
class Trust:
    evidence_anchor: str        # REQUIRED: stable structure outside N evidence sims
    blast_radius: str           # REQUIRED: local | workflow | global
    rollback_when: str          # REQUIRED: observable disabling condition
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE; optional otherwise
    fallback: str               # OPTIONAL: matcher-false / handler-allow semantics
```

`evidence_anchor` and `out_of_evidence_probe` are **load-bearing** for the durability audit. They are not free-text rationale:

- `evidence_anchor`: name a system field, a tool schema field, a protocol invariant, or a general algorithm. If your answer is "doc_015 says…" the structure you're anchored to is inside your evidence — pick a different class or do not write the component.
- `out_of_evidence_probe`: name one concrete case NOT in your evidence sims where your matcher would fire, and state what your handler returns on it. If you cannot construct one, the component overfits by construction.

### Capability

Explicit allowlist of side-effects the handler may perform. Declared at construction; v1 records it for audit, v2 will sandbox at fire time.

| capability        | what it permits                                                       |
|-------------------|-----------------------------------------------------------------------|
| `none`            | pure function; matcher + handler return values only                   |
| `read_file`       | `open(path, "r")` on workspace paths                                  |
| `http_get`        | outbound HTTP GET                                                     |
| `tool_call`       | issue a sub-tool-call within the handler (retriever pattern)          |
| `llm_call`        | invoke a sub-LLM within the handler (verifier pattern)                |
| `mutate_shared`   | write to `ctx.shared`                                                 |

## The class × mount × decision matrix

This is the load-time gate. Each cell lists permitted decisions; absent cells reject registration with `ComponentPolicyError`. Reproduced from `component_runtime/policy.py::ALLOWED`.

| class \ mount         | pre_context_build | session_start  | user_prompt_submit | pre_tool_use                       | post_llm_response                       | post_tool_use   | stop  |
|-----------------------|-------------------|----------------|--------------------|------------------------------------|-----------------------------------------|-----------------|-------|
| `mechanism_layer`     | inject_context    | inject_context | inject_context     | rewrite_tool_args, defer, block    | rewrite_tool_args, block, inject_context | inject_context | block |
| `reactive_guard`      | —                 | —              | inject_context     | block, rewrite_tool_args           | rewrite_tool_args, block, inject_context | inject_context | block |
| `channel`             | inject_context    | inject_context | inject_context     | —                                  | —                                       | —               | —     |
| `induced_rule`        | **inject_context (advisory)** | — | **inject_context (advisory)** | — | —                                       | —               | —     |
| `predictive_heuristic`| rejected at load  | rejected       | rejected           | rejected                           | rejected                                | rejected        | rejected |

A component whose (class, mount) pair is absent raises `ComponentPolicyError` at load. A handler that emits a non-admitted decision raises `ComponentPolicyError` at fire time. There is no convention to violate.

## Why these 4 admissible classes

`mechanism_layer`, `reactive_guard`, and `channel` carry their semantics from the hook system. The key change is **`induced_rule` REINSTATED but advisory-only**:

- Old hook system: `induced_rule` **rejected** at load time.
- New component system: `induced_rule` **admitted** but restricted to `mount ∈ {pre_context_build, user_prompt_submit}` and `decision = inject_context` only.

The concession: an `induced_rule` reading of a policy document, presented as an *advisory note* via `inject_context`, is strictly safer than the LLM rediscovering the same passage cold — the worst case is a redundant prompt. The LLM still does the final interpretation; the rule cannot mechanically override. This realises `RESULTS.md §6.4` ("give the LLM a sturdier shell rather than compile policy into code").

`predictive_heuristic` stays rejected because its matcher tests **prompt-shape** (regex on user-message content, etc.). Even an advisory injection on that activation conditions interpretation on raw text surface — structurally unsafe; no rewrite saves it.

The four admissible classes therefore partition by **what the matcher tests**:

- structure (system field / tool schema / protocol invariant / general algorithm) → `mechanism_layer`
- observed failure event → `reactive_guard`
- task structure (file present, URL mentioned, KB needs a doc) → `channel`
- policy reading → `induced_rule` (advisory only)
- prompt shape → no class fits; redesign or do not write

## The workflow graph model

The frontier is a typed graph, not a flat set.

- **Nodes:** named references to components by `Component.name`.
- **Edges:** v1 records `(src, dst)` pairs in YAML for visualisation; v1 dispatch does NOT use edges (ordering is `(priority, insertion)` per mount). v2 will honour edges for true graph traversal.
- **State on disk (TWO files, distinct purposes):**
  - `meta_harness/workflows/tau2_main.yaml` — canonical, human-editable graph. The outer loop mutates this in place after every iteration. You READ it; the outer loop writes it.
  - `meta_harness/logs_tau2_components/frontier_workflow.json` — JSON snapshot written ONLY at acceptance; the scoring loop's trust source for the "previous frontier" reference.

### YAML schema

```yaml
nodes:
  - close_account_strip_optional_reason
  - discoverable_audit_channel
edges: []                 # v1: declarative only
disabled: []
```

### The 3 patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; the file `agent_tau2/components/<id>.py` must be newly written                         |
| `replace_node` | keep the existing node id; overwrite the file with a new implementation (same `COMPONENT.name`)           |
| `disable_node` | add the existing node's id to `disabled:`; the file remains on disk for the durability audit              |

Sugar derivable from these (NOT exposed as separate ops):

- `wrap_tool(tool_name)` = `add_node` at `pre_tool_use` with a matcher testing `ctx.tool_name == tool_name`.
- `insert_before(node_id)` = `add_node` with edges_in pointing at the position. (v1 dispatch ignores edges; use `priority` to bias ordering.)

## Composition model

Each iteration proposes exactly one patch. The outer loop:

1. Reads `tau2_main.yaml`.
2. Applies your patch to produce a candidate graph.
3. Writes the new YAML, sets `COMPONENT_NAMES=<active>`, `COMPONENT_RUN_TAG=iter<N>`, runs `tau2_runner.py --candidate component_runtime` on train-30.
4. Compares the candidate's train-30 reward to the frontier reward.
5. If ≥ frontier: patch admitted; `frontier_workflow.json` snapshot written; component file kept; `.bak_iter<N>` deleted for `replace_node`. Otherwise: YAML reverted, new file unlinked for `add_node`, original restored from `.bak_iter<N>` for `replace_node`.

Component fires land in `.component-state/iter<N>/fired.jsonl` for the durability audit.

## Hard rules

- Exactly ONE patch per invocation. One of `add_node`, `replace_node`, `disable_node`.
- **You do NOT run the simulator.** No `tau2_runner.py`, no `tau2 run`. The outer loop scores.
- **No task-specific code.** No customer names, account / document ids, per-task branching, no encoding of gold answers or gold action sets.
- General documented policy may enter as **advisory context** via a `channel` (mounts `session_start` / `pre_context_build` / `user_prompt_submit`) or an `induced_rule` (mounts `pre_context_build` / `user_prompt_submit`, decision `inject_context` only). Compiling policy into a mechanical override is not admissible at any class.
- **The target inference model is LOCKED via the tau2 LLM config.** Components may NOT spin up a different model. `ctx.chat()` (Phase C+) IS permitted for sub-LLM verifier patterns — it routes through `agent.llm.chat` (locked SUT model name) with mutable inference params: `ctx.chat(messages, max_tokens=8192, temperature=0.0, system_override=None, tools=None)`. The helper does NOT accept a `model=` kwarg. Declaring `capabilities=(Capability.LLM_CALL,)` is required.
- For `replace_node`, the **first shell action** you take MUST be:
  ```bash
  cp agent_tau2/components/<existing>.py agent_tau2/components/<existing>.py.bak_iter<N>
  ```
  before overwriting the file. The outer loop relies on this `.bak` to roll back on reject.
- READ-ONLY paths: `tau2-bench-src/`, `tau2_runner.py`, `meta_harness/*`, `agent_tau2/v0/`, `agent_tau2/component_runtime/`, all earlier `agent_tau2/components/*.py` (you may only modify by `replace_node` which goes through the .bak protocol), all `meta_harness/logs_tau2_*/` directories EXCEPT `logs_tau2_components/`. The proposer harness physically hides forbidden paths during your run.
- You may NOT create a new agent directory. Your only file writes are: ONE component file under `agent_tau2/components/<name>.py` (+ its `.bak_iter<N>` for replace_node) and the `pending_eval.json` manifest. The workflow patch metadata lives INSIDE `pending_eval.json` — the outer loop applies it to `tau2_main.yaml`.

## Component file interface

```python
# agent_tau2/components/<name>.py
from __future__ import annotations

from agent_tau2.component_runtime.types import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, Mount, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Cheap predicate. Reads ctx.tool_name / ctx.tool_args /
    # ctx.incoming_message / ctx.history / ctx.domain_policy.
    # NEVER reads task_id (the runtime hides it).
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite_tool_args({...})  # or inject_context / block / allow / defer


COMPONENT = Component(
    name="<stable_component_id>",                  # reuse for replace_node
    cls=ComponentClass.MECHANISM_LAYER,
    mount=Mount.PRE_TOOL_USE,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    trust=Trust(
        evidence_anchor="<system field / tool schema / protocol invariant / general algorithm>",
        blast_radius="local",                      # local | workflow | global
        rollback_when="<observable condition for disabling>",
        out_of_evidence_probe="<concrete OOE case + handler return>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

Reference examples (already in the frontier):
- `agent_tau2/components/close_account_strip_optional_reason.py` — `mechanism_layer` / `pre_tool_use` / `rewrite_tool_args`
- `agent_tau2/components/discoverable_audit_channel.py` — `mechanism_layer` / `session_start` / `inject_context`

## Workflow

### 1. Understand the workflow (before any failure analysis)

Read the domain's policy document(s), the full tool catalog, and protocol invariants. Write a workflow summary in your builder log: the end-to-end happy path, mandatory orderings, prerequisite reads, KB structure, session-state requirements. Form hypotheses against this reference frame.

### 2. Read state and failed simulations

```
meta_harness/workflows/tau2_main.yaml                                the active graph
meta_harness/logs_tau2_components/frontier_workflow.json             frontier snapshot
meta_harness/logs_tau2_components/frontier_val.json                  per-task best
meta_harness/logs_tau2_components/evolution_summary.jsonl            one row per iter (incl. rejected)
meta_harness/tau2_train_task_ids.txt                                 30 tasks — your pool
agent_tau2/components/                                                component files on disk
.component-state/iter<K>/fired.jsonl                                  which components fired in iter K (preserved per iter)
```

Per-iter simulation dumps + summaries (preserved across iters; nothing is overwritten):

```
tau2-bench-src/data/simulations/tau2-runs/meta/v0__<domain>.json/results.json       v0 baseline / iter 0
tau2-bench-src/data/simulations/tau2-runs/meta/iter<K>/component_runtime__<domain>.json/results.json   iter K's candidate run
traces/v0__<domain>__summary.jsonl                                                  legacy v0 summary (if present)
traces/iter<K>__tau2_component_runtime__summary.jsonl                               iter K's summary jsonl
```

Each `results.json` carries `simulations[].messages` / `simulations[].reward_info` for every task. Pick 4-6 train tasks the frontier still fails (reward 0) and open the appropriate iter's `results.json`. Cross-reference `.component-state/iter<K>/fired.jsonl` to see which existing components fired in that iter (and which did not but should have). If `evolution_summary.jsonl` has any row with iter ≥ 1, also read those rows — each names `candidate.hypothesis` + `train_score` + `accepted`, so you can avoid re-proposing a mechanism a prior candidate already covered, and trace regressions back to the iter that introduced them.

### 3. Form ONE hypothesis

```
HYPOTHESIS:           <falsifiable claim about train-30 reward>
MECHANISM:            <failure mode> seen in N≥3 task simulations [tid1, tid2, ...]
STABLE STRUCTURE:     <evidence_anchor — system field / tool schema / protocol invariant /
                       general algorithm; must live OUTSIDE the N evidence sims>
OUT_OF_EVIDENCE PROBE: <one concrete case NOT in the evidence sims where the matcher fires,
                       and exactly what the handler returns on it>
PATCH_OP:             <add_node | replace_node | disable_node>
COMPONENT:            mount=<...>, cls=<...>, state_scope=<...>, capabilities=<...>
EXPECTED_DELTA:       train-30 reward <current> → <expected>
```

If you cannot name a STABLE STRUCTURE outside your evidence AND cannot answer OUT_OF_EVIDENCE PROBE concretely, no class admits your component (or `induced_rule` is the only fit — then your decision MUST be `inject_context` on a context-build mount, no override).

### 4. Classify class + choose mount

| matcher tests…                                                | class             |
|---------------------------------------------------------------|-------------------|
| system field / tool schema field / protocol invariant         | `mechanism_layer` |
| observed failure event in `ctx.incoming_message`              | `reactive_guard`  |
| task structure (file present, URL mentioned, KB doc needed)   | `channel`         |
| "the policy text says X applies to this case"                 | `induced_rule` (must use `inject_context`) |
| raw prompt text via regex/keywords                            | none — redesign or do not write |

Then choose the mount whose (class, mount) cell admits the decision you need.

### 5. Implement

Create exactly one file at `agent_tau2/components/<name>.py` (see `templates/component_file.py.template`). Validate the import:

```bash
python -c "
from agent_tau2.component_runtime.registry import load_components_from_dir
comps = load_components_from_dir(only=['<name>'])
ok = any(c.name == '<name>' for cs in comps.values() for c in cs)
print('component loads + passes policy:', ok)
"
```

For `replace_node`, FIRST run the `cp ... .bak_iter<N>` command (see Hard rules) before overwriting.

This is the ONLY shell command you run.

### 6. Choose patch op

| situation                                                                 | op             |
|---------------------------------------------------------------------------|----------------|
| no existing node addresses this mechanism                                 | `add_node`     |
| an existing node addresses this mechanism but has a known bug             | `replace_node` (reuse `COMPONENT.name`; .bak first) |
| an existing node is provably dead weight or actively harmful              | `disable_node` (no new file; reference existing id) |

### 7. Write `pending_eval.json`

See `templates/pending_eval.json.template`. Schema:

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "component_iter<N>_<slug>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected reward change on train-30>",
    "component": {
      "id": "<stable_component_id>",
      "cls": "mechanism_layer | reactive_guard | channel | induced_rule",
      "mount": "pre_context_build | session_start | user_prompt_submit | pre_tool_use | post_llm_response | post_tool_use | stop | session_end",
      "state_scope": "none | session | cross_session",
      "capabilities": ["none | read_file | http_get | tool_call | llm_call | mutate_shared"],
      "file": "agent_tau2/components/<name>.py",
      "trust": {
        "evidence_anchor": "<stable structure outside evidence sims>",
        "blast_radius": "local | workflow | global",
        "rollback_when": "<observable rollback condition>",
        "out_of_evidence_probe": "<concrete OOE case + handler return>",
        "fallback": "<matcher-false / handler-allow semantics>"
      }
    },
    "workflow_patch": {
      "op": "add_node | replace_node | disable_node",
      "name": "<component name; existing id for disable_node>",
      "file": "agent_tau2/components/<name>.py",
      "edges_in": [],
      "edges_out": []
    }
  }
}
```

For `disable_node`, the `component` block can be omitted (or left empty); only `workflow_patch.name` matters.

### 8. Session log + exit

Write a concise log to `meta_harness/logs_tau2_components/builder_sessions/iter<N>/log.md` using `templates/hypothesis.md.template`. Final line of your reply:

```
CANDIDATE: component_iter<N>_<slug>
```

## Common patterns

| pattern                                | class             | mount                | decision                  | example                                                |
|----------------------------------------|-------------------|----------------------|---------------------------|--------------------------------------------------------|
| strip an optional arg                  | mechanism_layer   | pre_tool_use         | rewrite_tool_args         | `close_account_strip_optional_reason`                  |
| inject a framework constant            | mechanism_layer   | session_start        | inject_context            | `discoverable_audit_channel`                           |
| retrieve a KB doc per session          | channel           | pre_context_build    | inject_context            | retriever with `capabilities=(TOOL_CALL,)`             |
| wrap one tool's args                   | mechanism_layer   | pre_tool_use         | rewrite_tool_args         | matcher uses `matcher_for_tool("close_bank_account")`  |
| react to `tool.failed`                 | reactive_guard    | post_tool_use        | inject_context            | append a "retry with X" note for next turn             |
| block a malformed tool call            | reactive_guard    | pre_tool_use         | block                     | observed mid-turn shape error                          |
| sub-LLM verifier on assistant turn     | mechanism_layer   | post_llm_response    | rewrite_tool_args / block | self-consistency check before dispatch                 |
| advisory policy note                   | induced_rule      | pre_context_build    | inject_context            | "the closure flow has prerequisites; consult doc_021"  |
| escalation gate                        | reactive_guard    | stop                 | block                     | ensure escalation steps not skipped (reserved)         |

See `patterns/<mount>.md` for per-mount worked examples.

## Event runtime additions (Phase D)

The mount-based dispatch above remains the canonical entry — every existing
component fires via `listens = mount.value` by default (auto-set in
`Component.__post_init__`). Phase D adds a parallel **Tier-1 event** namespace.
In tau2 the legacy `_dispatch_pre_tool_use` / `_dispatch_post_llm_response` /
`_dispatch_post_tool_use` paths are unchanged (they own tau2-shaped
`ToolCall` / `AssistantMessage` rewriting); the Tier-1 events fire through a
parallel core dispatcher that uses the same component set but a coarser
INJECT_CONTEXT / BLOCK applier.

### Tier-1 events emitted by `ComponentLLMAgent` (parallel to mount dispatch)

| event                       | when it fires                                                         | mount alignment             |
|-----------------------------|-----------------------------------------------------------------------|-----------------------------|
| `task_received`             | start of `_collect_prompt_injection` (before any PRE_CONTEXT_BUILD)    | (new — lifecycle anchor)    |
| `pre_context_build`         | paired with `Mount.PRE_CONTEXT_BUILD`                                  | = `Mount.PRE_CONTEXT_BUILD` |
| `session_start`             | static system_prompt injection                                          | = `Mount.SESSION_START`     |
| `pre_agent_construct`       | end of `_collect_prompt_injection` (instructions sealed)               | (new)                       |
| `pre_llm_request`           | before `super().generate_next_message` per turn                        | (new)                       |
| `post_llm_response_raw`     | after `super().generate_next_message` returns                          | aliases `post_llm_response` |
| `on_empty_response`         | **synthesised** when assistant content is empty + no tool calls        | (new)                       |
| `on_no_tool_call_emitted`   | **synthesised** when no tool_calls in assistant turn                   | (new)                       |
| `pre_tool_arg_validation`   | per ToolCall, **before** `_dispatch_pre_tool_use` loop                 | (new)                       |
| `pre_tool_use`              | per ToolCall, via legacy mount path                                    | = `Mount.PRE_TOOL_USE`      |
| `post_llm_response`         | per AssistantMessage, via legacy mount path                            | = `Mount.POST_LLM_RESPONSE` |
| `post_tool_result_raw`      | per ToolMessage in `_dispatch_post_tool_use`                           | (new)                       |
| `on_tool_error`             | **synthesised** when `tm.error` is set                                 | (new)                       |
| `post_tool_use`             | per ToolMessage, via legacy mount path                                 | = `Mount.POST_TOOL_USE`     |
| `on_explicit_terminate` / `session_end` | reserved (not yet emitted in v1)                                      | (deferred)                  |

INJECT_CONTEXT decisions from Tier-1 subscribers land either in
`ctx.shared['tier1_prompt_inject']` (pre-LLM events, merged into the next
instructions extension) or `ctx.shared['post_llm_inject']` (post-event,
queued for the next outer turn as a SystemMessage).

### `Component.listens` and `emits` (Phase B)

```python
COMPONENT = Component(
    ...,
    listens="on_tool_error",          # default = mount.value; existing
                                       # mount-only components migrate transparently.
    emits=("iter12_tool_retry_planned",),  # self-doc of custom events.
)
```

The `mount` field is still required for policy validation. Pick the Mount
enum that conceptually owns the event; the dispatcher buckets by `listens`.

### `ctx.chat()` / `ctx.emit()` / `ctx.emit_upstream()` (Phase C)

`ComponentContext` inherits from `EventContext`:

| method                                | semantics                                                                                          |
|---------------------------------------|----------------------------------------------------------------------------------------------------|
| `ctx.chat(messages, *, max_tokens=8192, temperature=0.0, system_override=None, tools=None)` | Sub-LLM call bound to the locked SUT model name. No `model=` kwarg.                                  |
| `ctx.emit(custom_event_name, **fields)` | Synchronously fire a Tier-2/3 custom event through the same core dispatcher. Depth cap = 10. Fields dropped; use `ctx.shared` / `ctx.upstream`. |
| `ctx.emit_upstream(key, value)`       | `ctx.upstream[key] = value` sugar.                                                                  |
| `ctx.fetch` / `ctx.read_file`         | None-wired in tau2 v1.                                                                              |

Declare `capabilities=(Capability.LLM_CALL,)` for components that use `ctx.chat()`.

### Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter12_audit_visibility_resolved
on_<thing>                    failure-mode style
```

Always declare `emits=(...)` on the publisher; grep
`agent_tau2/components/*.py` for taken names. To admit non-ALLOW decisions on
a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in
`agent_tau2/component_runtime/policy.py`.

### Event-based patterns

| pattern                                              | class           | listens                    | decision                                    |
|------------------------------------------------------|-----------------|----------------------------|---------------------------------------------|
| schema-check tool args narrowly                      | mechanism_layer | `pre_tool_arg_validation`  | rewrite_tool_args (via `ctx.tool_call` rewrite) |
| retry hint on observed tool error                    | reactive_guard  | `on_tool_error`            | inject_context (queued for next turn)       |
| sub-LLM verifier on final assistant turn              | mechanism_layer | `post_llm_response_raw`    | rewrite_tool_args / block (via `ctx.chat`)  |
| publisher/subscriber dataflow within one task        | mechanism_layer | (A emits `iter<N>_<slug>_X`) → (B `listens="iter<N>_<slug>_X"`) | inject_context (consume `ctx.upstream`)      |

## What this skill does NOT do

- Run the tau2 simulator (the outer loop does).
- Modify `tau2-bench-src/`, the eval, the model, `tau2_runner.py`, `meta_harness/`, `agent_tau2/v0/`, `agent_tau2/component_runtime/`, prior component files (except via `replace_node` + `.bak` protocol).
- Build a full `LLMAgent` subclass (that pattern was retired in favour of components).
- Build a hook (the hook system was retired; components subsume it).
- Task-specific hardcoding — encoding gold answers or branches keyed to specific task ids.
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as a `mechanism_layer` override (`rewrite_tool_args` / `block`). The matrix routes such proposals to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as a `predictive_heuristic` — that class is rejected at load.
