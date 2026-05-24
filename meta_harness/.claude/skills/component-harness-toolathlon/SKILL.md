---
name: component-harness-toolathlon
description: Run ONE iteration of Toolathlon (hkust-nlp/Toolathlon) harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/toolathlon_main.yaml`. The component is typed by mount + class + state_scope, gated by a class×mount×decision permission matrix, and verified by a Trust block (evidence_anchor + out_of_evidence_probe). Sibling of component-harness-gaia / -tau2 / -sopbench; targets the OpenAI Agents SDK + MCP gateway loop that Toolathlon ships, with a smaller dispatched-mount set than tau2 due to SDK hook surface limits.
---

# component-harness-toolathlon

Run ONE iteration of agent evolution against Toolathlon's task corpus by proposing ONE **workflow graph patch**. A patch is `add_node` / `replace_node` / `disable_node`, applied to the frontier workflow at `meta_harness/workflows/toolathlon_main.yaml`. The node added or replaced is one **component** — a single Python file in `agent_toolathlon/components/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior baselines + traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on the train subset (28 task ids), and admits or rejects.

## What Toolathlon tasks look like

A task = one directory under `Toolathlon-src/tasks/finalpool/<task_id>/`. The agent receives:
  * a natural-language user instruction (the `task_str` — what the user wants done)
  * a set of MCP servers exposing tools (filesystem, GitHub, Notion, GCP, arxiv-local, scholarly, browser, etc.)
  * a Python-based per-task verifier that runs after the agent finishes

The agent runs an OpenAI Agents SDK loop with MCP tools and (for multi-turn tasks) a user simulator. **The verifier is binary**: pass / fail. Failure mechanisms vary widely:

  * **wrong final emission** — model wrote a passable answer but in the wrong format (Markdown when plain text was asked, extra commentary).
  * **missing required tool call** — model answered from prior knowledge without invoking the prescribed lookup (e.g. didn't actually `arxiv_local-download_paper`).
  * **tool-arg hallucination** — model passed `paper_id="2505.20286v1"` when the tool requires `paper_id="2505.20286"` (versionless), or invented filesystem paths outside `/workspace/dumps/workspace`.
  * **prompt-instruction omission** — model ignored part of the multi-part user request (e.g. asked for title + abs_url + code_url, model returned only title).
  * **scoping mistake** — model touched the wrong Notion page / GitHub repo because it didn't read the verifier's allowlist.
  * **runaway tool loop** — model retried the same MCP call dozens of times when it kept failing (no fallback strategy).

Your job is to pick ONE such mechanism present in ≥3 train failures and add ONE component that addresses it.

## First principles

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — system-prompt scaffolding, output format constraints, tool-arg shape validation reminders, retry-on-loop heuristics — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split: what is the minimal judgment the LLM must make, and what computation / lookup / validation around it is fully deterministic? Build the deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact OUTSIDE evidence — a tool's declared JSON Schema, an SDK API field, an MCP server's documented behavior, the user-instruction grammar (e.g. "return X, Y, Z in this exact format"). An IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Prefer mounts that work in single-turn mode.** 90% of Toolathlon tasks run with `single_turn_mode=True`; the user simulator never re-prompts. PRE_CONTEXT_BUILD / SESSION_START / USER_PROMPT_SUBMIT all reach the LLM even in single-turn. POST_TOOL_USE INJECT_CONTEXT is QUEUED for the next outer user turn, which in single-turn tasks NEVER comes. Design accordingly.

## v2 mount semantics (read this carefully)

Toolathlon v2 wraps every MCP tool as an SDK FunctionTool before handing it to the Agent. This lets the component runtime intercept BEFORE and AFTER the real tool invocation with the actual arguments and the actual result string — solving the single-turn limitations of v1. (Set `COMPONENT_WRAP_TOOLS=0` to fall back to v1 path; not recommended.)

| mount               | when it fires                                                 | dispatched? | what works                                                                                                                                                       |
|---------------------|---------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pre_context_build` | once per task, before Agent() is constructed                   | ✅           | INJECT_CONTEXT → spliced into Agent.instructions                                                                                                                 |
| `session_start`     | once per task, after PRE_CONTEXT_BUILD                         | ✅           | INJECT_CONTEXT → spliced into Agent.instructions (after PRE_CONTEXT_BUILD's contributions)                                                                       |
| `user_prompt_submit`| each outer user turn, after user_query obtained                | ✅           | INJECT_CONTEXT → appended to the user message before it lands in logs                                                                                            |
| `pre_tool_use`      | inside FunctionTool wrapper, before MCP `call_tool`            | ✅           | ALLOW; **REWRITE_TOOL_ARGS** (real-args-aware, sequential composition across components); **true BLOCK** (tool is NOT called; component-block string returned). DEFER still rejected (v2.5). |
| `post_llm_response` | SDK has NO mid-turn AssistantMessage hook                      | ❌ REJECTED  | (registration fails at load time — would need wrapping ModelProvider; v3.)                                                                                       |
| `post_tool_use`     | inside FunctionTool wrapper, after MCP `call_tool`             | ✅           | INJECT_CONTEXT is **concatenated INTO the tool result string**, so the LLM sees it on its very next inference — works in single-turn AND multi-turn.             |
| `stop`              | reserved                                                       | ❌ not dispatched yet | (registration allowed; runtime ignores)                                                                                                                          |
| `session_end`       | reserved                                                       | ❌ not dispatched yet | (registration allowed; runtime ignores)                                                                                                                          |

**Practical guidance:**

* **PRE_TOOL_USE REWRITE_TOOL_ARGS / BLOCK and POST_TOOL_USE INJECT_CONTEXT now ACTUALLY WORK in single-turn tasks.** They are no longer multi-turn-only as in v1.
* **REWRITE_TOOL_ARGS composes**: if multiple components match the same tool, they fire in `(priority, insertion)` order and each sees the previous component's rewritten args.
* **BLOCK short-circuits**: the MCP `call_tool` is skipped entirely; the tool result handed back to the LLM is a `<component_block …>` string. The LLM sees that and can decide what to do next.
* **DEFER is still rejected.** It needs a replay queue (post-condition re-invocation) which v2.5 will add.
* **POST_LLM_RESPONSE is still rejected.** Wrapping ModelProvider is the v3 path.
* PRE_CONTEXT_BUILD / SESSION_START / USER_PROMPT_SUBMIT still work in every task as in v1; prefer them when the failure mode is "wrong system prompt / wrong user instruction interpretation".

## The component model

```python
@dataclass(frozen=True)
class Component:
    name: str
    cls: ComponentClass
    mount: Mount
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100
```

### ComponentClass enum

| class                 | the matcher tests…                                                                                                                                                                       | risk                          |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`     | a system field, an MCP tool-declared JSON Schema, an SDK lifecycle property, a general algorithm (URL parse, version stripping, regex on tool name)                                       | LOW                           |
| `reactive_guard`      | an observed failure event (tool error string contains "denied", missing required substring in final response, repeated identical tool call indicating a loop)                              | LOW                           |
| `channel`             | task structure — instruction text grammar ("return X, Y, Z"), allowed-MCP-list, presence/absence of a server in `task_config.needed_mcp_servers`                                          | LOW                           |
| `induced_rule`        | a reading of policy/instruction text — IF/THEN compiled from N evidence rows                                                                                                              | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic`| raw prompt / response text guessing what the model is about to do                                                                                                                         | REJECTED at load time         |

### Decision

| decision         | semantics in toolathlon v1                                                                                                                                                                                |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                                                                                                                                       |
| `block`          | PRE_TOOL_USE (v2): the MCP `call_tool` is SKIPPED; the tool result handed to the LLM is a `<component_block …>` string. STOP: reserved.                                                                     |
| `inject_context` | PRE_CONTEXT_BUILD / SESSION_START → appended to `Agent.instructions`; USER_PROMPT_SUBMIT → appended to user_query before it lands in logs; POST_TOOL_USE (v2) → CONCATENATED into the tool result string.    |
| `rewrite_tool_args` | PRE_TOOL_USE (v2): the dict you return REPLACES the LLM-emitted args before the MCP call. Multiple components compose left→right.                                                                            |
| `defer`          | **REJECTED at load time** in v2 (replay queue lands in v2.5).                                                                                                                                                |

### StateScope

`none` (default) / `session` (per-task scratchpad at `ctx.state[component_name]`) / `cross_session` (reserved).

### Trust

```python
@dataclass(frozen=True)
class Trust:
    evidence_anchor: str        # REQUIRED — name the stable structure being targeted
    blast_radius: str           # REQUIRED: local | workflow | global
    rollback_when: str          # REQUIRED — concrete signal that says "stop using this"
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE
    fallback: str               # OPTIONAL
```

### Capability

`none` / `read_file` / `http_get` / `llm_call` / `tool_call` / `mutate_shared`.

## The class × mount × decision matrix (toolathlon v2)

Load-time gate at `agent_toolathlon/component_runtime/policy.py::ALLOWED`.

| class \ mount         | pre_context_build | session_start | user_prompt_submit | pre_tool_use            | post_llm_response | post_tool_use   | stop          | session_end |
|-----------------------|-------------------|---------------|--------------------|-------------------------|-------------------|-----------------|---------------|-------------|
| `mechanism_layer`     | allow, inject     | allow, inject | allow, inject      | allow, block, rewrite   | **rejected**      | allow, inject   | allow, block  | allow       |
| `reactive_guard`      | —                 | —             | allow, inject      | allow, block, rewrite   | **rejected**      | allow, inject   | allow, block  | allow       |
| `channel`             | allow, inject     | allow, inject | allow, inject      | —                       | **rejected**      | —               | —             | —           |
| `induced_rule`        | allow, inject     | —             | allow, inject      | —                       | **rejected**      | —               | —             | —           |
| `predictive_heuristic`| rejected          | rejected      | rejected           | rejected                | rejected          | rejected        | rejected      | rejected    |

A component whose (class, mount) is absent raises `ComponentPolicyError` at load. A handler that emits a non-admitted decision raises at fire time. `post_llm_response` is rejected for ALL classes in v2 (would need v3 ModelProvider wrapping). `defer` is rejected for ALL classes in v2 (would need v2.5 replay queue).

## The workflow graph

Single frontier yaml: `meta_harness/workflows/toolathlon_main.yaml`

```yaml
nodes:
  - some_component_name
edges: []
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_toolathlon/frontier_workflow.json` written on accept.

### Patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent_toolathlon/components/<id>.py` must be newly written                            |
| `replace_node` | keep the existing node id; overwrite the file with new behavior (same `COMPONENT.name`)                   |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                          |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `toolathlon_runner.py`. The outer loop scores.
- **No task-specific code.** No train task_ids in matchers. No hardcoded entity strings (paper IDs, GitHub repo names, Notion page slugs). No encoded gold answers.
- **The target inference model (deepseek-v4-pro via Together AI) is LOCKED.** It is the System Under Test. Do NOT call any other LLM API from a component; do NOT attempt to override the model in `toolathlon_runner.py`. Components must be deterministic Python (or call the SDK only via the dispatched mount — never a raw chat API).
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent_toolathlon/components/<existing>.py agent_toolathlon/components/<existing>.py.bak_iter<N>
  ```
  The outer loop relies on the `.bak` to roll back on reject.
- READ-ONLY (do not modify): `Toolathlon-src/`, `bench/toolathlon/`, `toolathlon_runner.py`, `agent_toolathlon/runtime/`, `agent_toolathlon/v0/`, `agent_toolathlon/cr/`, `agent_toolathlon/component_runtime/` (the runtime itself is locked; you only write to `agent_toolathlon/components/`). Also: `meta_harness/meta_harness_components_toolathlon.py`, `meta_harness/toolathlon_*.txt`.
- Component file naming: `agent_toolathlon/components/component_iter<N>_<slug>.py`. The `COMPONENT.name` should follow the same `component_iter<N>_<slug>` pattern (helps audit which iter introduced it).

## Component file template

```python
# agent_toolathlon/components/component_iter<N>_<slug>.py
from __future__ import annotations

from agent_toolathlon.component_runtime.types import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, Mount, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.tool_name (PRE_TOOL_USE / POST_TOOL_USE), ctx.tool_args (= {} on
    # PRE_TOOL_USE in v1 — SDK doesn't surface args), ctx.incoming_message
    # (POST_TOOL_USE: dict with tool_name + output string), ctx.history
    # (snapshot of self.logs), ctx.tool_names (full tool list), ctx.shared.
    # ctx.proposed_system_prompt is the in-progress prompt during PRE_CONTEXT_BUILD.
    #
    # NEVER read or compare against task_id-specific data — that is memorisation.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context("...")   # or allow / block


COMPONENT = Component(
    name="component_iter<N>_<slug>",
    cls=ComponentClass.MECHANISM_LAYER,
    mount=Mount.PRE_CONTEXT_BUILD,
    matcher=_matches,                       # or None for always-on
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    trust=Trust(
        evidence_anchor="...",      # what stable structure is this targeting?
        blast_radius="local",       # local | workflow | global
        rollback_when="...",        # how would you know to disable this?
        # out_of_evidence_probe="..." if INDUCED_RULE
    ),
)
```

## `pending_eval.json` schema

The outer loop reads this file after your run. Write it once, validated.

```json
{
  "candidate": {
    "name": "mh_toolathlon_iter<N>_<slug>",
    "hypothesis": "one-sentence claim of what the failure mode is",
    "changes": "one-sentence description of what your component does",
    "component": {
      "name": "<COMPONENT.name>",
      "cls": "mechanism_layer",
      "mount": "pre_context_build",
      "file": "agent_toolathlon/components/component_iter<N>_<slug>.py",
      "trust": {
        "evidence_anchor": "...",
        "blast_radius": "local",
        "rollback_when": "...",
        "out_of_evidence_probe": "",
        "fallback": ""
      }
    },
    "workflow_patch": {
      "op": "add_node",
      "name": "<COMPONENT.name>",
      "file": "agent_toolathlon/components/component_iter<N>_<slug>.py",
      "edges_in": [],
      "edges_out": []
    }
  }
}
```

For `disable_node`, omit `file` and the only `name` is the existing component's name. For `replace_node`, `name` reuses the existing `COMPONENT.name`; `file` is the same path you overwrote (registry uses last-loaded-wins by name).

## How to investigate

State files and trace locations (read-only; the outer loop writes them):

```
meta_harness/workflows/toolathlon_main.yaml                            active workflow graph
meta_harness/logs_components_toolathlon/frontier_workflow.json         frontier snapshot
meta_harness/logs_components_toolathlon/frontier_val.json              per-task best for the CURRENT frontier (= last accepted candidate)
meta_harness/logs_components_toolathlon/evolution_summary.jsonl        one row per iter (incl. rejected); each row's per_task[*].dump_dir points at THAT iter's run
.component-state-toolathlon/toolathlon_iter<K>/fired.jsonl             which components fired in iter K (preserved per iter)
```

Per-task trace directories (preserved across iters; nothing is overwritten):

```
Toolathlon-runs/v0/finalpool/<tid>/                          v0 baseline / iter 0
Toolathlon-runs/cr/iter<K>/finalpool/<tid>/                  iter K's candidate run (accepted OR rejected)
Toolathlon-runs/cr/final_test/finalpool/<tid>/               held-out test pass (only after evolution finishes)
```

Each `<tid>/` directory holds `traj_log.json`, `eval_res.json`, `host_loop.log`. The `evolution_summary.jsonl` row for iter K already has `per_task[*].dump_dir` filled with the correct `cr/iter<K>/...` path — you don't have to construct it.

1. **Read `frontier_val.json`'s `per_task` map.** Each entry has `passed`, `tier`, `dump_dir`, `error`. This is the CURRENT frontier — start here to see what the in-place candidate still fails.
2. **If `evolution_summary.jsonl` has any row with iter ≥ 1, also read those rows.** Each one names a `candidate.hypothesis` + `train_score` + `accepted` + `per_task`, and its `per_task[*].dump_dir` points at the actual `cr/iter<K>/finalpool/<tid>/` trace from that iter. Compare a prior row's per_task against the v0 / frontier per_task to see which tasks that candidate broke (regression = was passing before, failing in iter K). Avoid re-proposing a mechanism a prior `hypothesis` already covered.
3. **For each failing task_id (`passed=false`)**, read the three trace files at `<dump_dir>/`:
   * `traj_log.json` — full message log (user / assistant / tool messages), `config.single_turn_mode`, `key_stats`, `status` (success / failed / max_turns_reached / interrupted).
   * `eval_res.json` — `{pass: bool, details: str}` from the per-task verifier (often the most informative single file).
   * `host_loop.log` — pretty-printed TOOL_CALL / TOOL_OUT / SUMMARY events in chronological order.
4. **Look for ≥3 failures sharing the same mechanism.** Examples:
   - 3 failures end with the assistant returning a Markdown-formatted answer when the instruction asked for "plain text without markdown" → `pre_context_build` INJECT a strict-format reminder.
   - 3 failures call `gw-arxiv_local-read_paper` with `paper_id="2505.20286v1"` (with version) when the corpus only has versionless ids → `user_prompt_submit` INJECT a hint about arxiv id versionless form, OR `session_start` INJECT a behavioral rule about MCP tool argument shape.
   - 3 failures retry the same failing tool >5 times in a row (eval_res shows max_turns_reached) → `session_start` INJECT a "if a tool fails the same way 3 times, switch strategy" instruction (MECHANISM_LAYER, anchored on the general algorithmic principle that repeated identical failures indicate stuck-state).
5. **Form ONE hypothesis** and tie it to a stable structure (an MCP tool's schema; the user instruction's grammar; an SDK status field; an OpenAI API field). State the structure in `trust.evidence_anchor`.
6. **Write ONE component** at the appropriate mount with the smallest possible matcher/handler. Resist embedding task-specific entity names.
7. **Validate registration**:
   ```bash
   python -c "
   import sys; sys.path.insert(0, '.')
   from agent_toolathlon.component_runtime.registry import load_components_from_dir
   comps = load_components_from_dir('agent_toolathlon/components',
                                     only=['<COMPONENT.name>'])
   assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
   print('ok')
   "
   ```
8. **Write `pending_eval.json`** and print `CANDIDATE: <name>`.

## What NOT to write

- Components that memorise train-set answers (encode "if task instruction contains 'Alita', return paper_id=2505.20286").
- Components that try to compute the final answer deterministically without calling the LLM (defeats the SUT measurement).
- Components that touch `toolathlon_runner.py`, `Toolathlon-src/`, `agent_toolathlon/component_runtime/`, or `agent_toolathlon/runtime/`.
- Components that need POST_LLM_RESPONSE or DEFER — load-time rejected in v2 (v2.5 / v3).
- Components that fire on EVERY task (priority=0 with always-True matcher) — that's effectively a prompt rewrite, not a component. Use a matcher that anchors on a stable structural predicate.
