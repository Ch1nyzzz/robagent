# Open questions — robagent next-step design

A snapshot of the active design question as of 2026-05-23. See `RESULTS.md`
for what's already been measured; this file is what's still undecided.

## Where the project is

We have two sibling skills under `meta_harness/.claude/skills/`:

- **`robust-harness-tau2/`** — every candidate is a full `LLMAgent` subclass
  under `agent_tau2/mh_tau2_iter*/`. One candidate = one monolithic agent;
  two deterministic interventions cannot stack in one run. Used through
  `meta_harness/meta_harness_tau2.py`.
- **`hook-harness-tau2/`** — every candidate is a single Python **hook**
  file under `agent_tau2/hooks/<name>.py` exporting `HOOK: Hook`. Hooks
  attach to one of 6 declared lifecycle events of the base `LLMAgent` and
  the frontier is the **set** of accepted hook names (`frontier_hooks.json`).
  Used through `meta_harness/meta_harness_hooks.py`.

The hook system has a runtime under `agent_tau2/hook_runtime/`:

- `types.py` — `Hook`, `HookEvent`, `HookClass`, `Decision`, `HookContext`
- `policy.py::ALLOWED` — class × event × decision-kind permission matrix,
  validated at registration and at fire time (`HookPolicyError`)
- `registry.py` — load hook modules from a directory, dedup by `HOOK.name`
- `agent.py` — `HookedLLMAgent` wraps stock `LLMAgent`; v1 dispatches
  `session_start` / `pre_tool_use` / `post_tool_use` only (other 3 events
  declared and validated but not yet dispatched)

The outer loop `meta_harness/meta_harness_hooks.py` wraps the proposer call
in `_isolate_legacy(ROOT)` — a context manager that physically renames
`agent_tau2/mh_tau2_iter*/`, `meta_harness/logs_tau2*/` (except
`logs_tau2_hooks`), and two iter-viewpoint memory files into a timestamped
stash for the duration of the run; restores on exit. This is to keep the
hook frontier from being seeded by knowledge of the robust frontier.

Two hooks live under `agent_tau2/hooks/` today, both `deterministic_glue`:
- `close_account_strip_optional_reason.py` (hand-written example,
  `pre_tool_use`)
- `hook_iter1_discoverable_audit_channel.py` (produced by one
  `--proposer-only` invocation, `session_start`)

Neither is in `frontier_hooks.json` yet (no full eval pass has been run
under the hook outer loop).

## The open question

**Hooks are not the full shape of a harness.** A hook is a synchronous,
turn-local interceptor on the agent loop. It cannot, by construction:

- maintain cross-session state / indexes
- spawn independent sub-LLM inference
- change the tool catalog itself (add / hide / decorate tools)
- change inference-time parameters (model, thinking budget, sampling)

If the project's stated goal — "evolve from raw LLM to the best stable
harness possible" — is taken seriously, hooks cover at most one layer of
that harness. We need a small set of **orthogonal high-level component
types**, each with:

- its own runtime / interface contract
- its own permission matrix (the analogue of `hook_runtime/policy.py`)
- its own sibling proposer skill

so that the meta-loop can propose any of them per iteration, not just hooks.

## Current proposal (in flux)

Survey of how Anthropic / OpenAI / open-source frameworks group agent
harness components (Claude Code 5-layer, OpenAI Agents SDK 6-primitive,
MindStudio 9-component, ETCLOVG 7-layer, awesome-harness-engineering
12-primitive, DSPy modules, gerl.dev 6-class taxonomy). Cross-framework
consensus collapses to ~4 structural component types when classified by
**how the component interfaces with the LLM**, not by use-case domain
(retrieval / memory / planning are *use-cases*, not types):

| # | Type | Boundary to the LLM | LLM-facing? |
|---|---|---|---|
| 1 | **Hook** | intercepts LLM input / output events on the agent loop | invisible |
| 2 | **Tool** (incl. retrieval / validator / computed / composite / transform sub-types) | a callable in the LLM's tool catalog | visible (schema) |
| 3 | **Sub-Agent** | another independent-context LLM instance | invoked as a tool or via handoff |
| 4 | **Inference Config** | LLM call parameters themselves (model, sampling, thinking budget) | indirect |

Plus two non-main-path support types:
- **Observer** — collects metrics, never affects behaviour
- **Meta-Controller** — decides which version of each of the above runs

**Why retrieval / memory are not separate types**: an indexed retrieve
*looks* like an independent subsystem, but the only thing the LLM sees
is a tool schema. Index build, cross-session cache, embedding pipeline
are implementation details of that tool. Grouping them as "Tool
(retrieval sub-type)" keeps the classification cut on structural
boundaries, not feature areas.

### Mapping our 21 memory facts to the proposal

| memory mechanism | proposed type |
|---|---|
| `close_account_strip_optional_reason`, `discoverable_audit_channel` | Hook |
| `card_spec_channel`, `retention_protocol_lookup`, `cashback_audit` | Tool (retrieval) |
| `provisional_credit_rule`, `eligible_for_dispute` | Tool (validator) |
| `card_last_4_resolver` | Tool (computed) |
| `cashback_policy_engine` | Sub-Agent (verifier) or Tool (composite) |
| `write_ordering_controller (defer + replay)` | Hook (needs v2 DEFER dispatch) |

So the next sibling skill to build is **Tool**, not "retriever" — it
absorbs at least the four memory mechanisms above.

## What's actually unsettled

The user has indicated more input is coming on this classification
before we finalise. Specifically still undecided:

1. Is the 4-type cut correct, or does it need a 5th type / different cut?
2. Per-type implementation priority order
3. Whether the Tool sub-types (retrieval / validator / computed / composite
   / transform) share one Tool skill, or whether some deserve their own
   sibling skill
4. Whether Observer and Meta-Controller deserve elevation to first-class
   types
5. Whether `induced_rule` / `predictive_heuristic` should be reinstated
   when (4) is settled — currently deleted from the Hook system's enum,
   but they may have a place in Tool or Sub-Agent (where they could be
   structurally constrained differently)

Until (1)–(5) are settled, only the Hook skill (`hook-harness-tau2`) is
implemented. The Robust skill (`robust-harness-tau2`) remains the
catch-all for everything that doesn't fit Hook today.
