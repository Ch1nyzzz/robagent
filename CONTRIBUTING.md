# Contributing to Ballast

Ballast is a research project that shares a design philosophy. Contributions are
welcome, but the philosophy is the point — a change that wins on a train split by
violating it is not a contribution we can take. Read the four principles in
`README.md` first.

## Two kinds of change

**1. A stabilization hook** (the common case).
A hook stabilizes the main agent on a recurring failure mode. It is one Python
file declaring one `Component` that `listens` to a lifecycle event and returns a
`Decision`. Before writing one, answer honestly:

- *Is this stabilization or capability?* If the agent fails because it cannot
  *reach* something, that is a capability gap — register a tool instead. Hooks
  only stabilize what the agent can already do.
- *What stable structure does it anchor on?* A system field, an LLM API field
  (`finish_reason`), a tool's JSON schema, a general algorithm. If the only
  anchor is "I saw traces 17/23/41 do X," it is memorization — redesign or drop
  it. The load-time `class × event × decision` matrix will reject the unsafe
  forms (`predictive_heuristic`, mechanical `induced_rule` override) anyway.

The per-benchmark proposer contract under `ballast/.claude/skills/<name>/SKILL.md`
is the authoritative spec for the hook model, the lifecycle events, the Trust
block, and the validation step. It is the same contract the automated proposer
follows.

**2. A new benchmark adapter** (capability of the harness itself).
A new validation surface needs: a candidate runner (a thin adapter, like
`run_benchmark.py`), an outer-loop driver (`ballast/evolve_<name>.py`), a
lifecycle runtime, and a proposer `SKILL.md`. Do **not** vendor the upstream
dataset or simulator — depend on it via clone/install and keep it out of the
tree (see `.gitignore`).

## The experiment record is the artifact

Each benchmark keeps a lightweight, checked-in record under
`ballast/logs_components_<name>/`:

- `frontier_val.json` / `frontier_workflow.json` — the accepted frontier
- `evolution_summary.jsonl` — one row per iteration, **including rejected ones**

Everything else a run produces (proposer sessions, event traces, `logs/`,
`traces/`, large run outputs) is regenerable and is git-ignored. When you submit
an accepted change, commit the updated record alongside the hook so the result is
reproducible from the repo.

## Ground rules

- **One atomic change per iteration.** Don't bundle.
- **No task-specific code.** No entity names, no per-task branching, no encoded
  gold answers, no matcher keyed on `task_id`.
- **The target inference (SUT) model is locked.** Hooks may not override it.
- **Verify before you commit.** A change must load (pass the policy matrix) and
  must not regress train accuracy versus the prior frontier.

## Workflow

1. Fork and branch.
2. Make your change; run the relevant baseline / evolution command from
   `README.md` to confirm it loads and does not regress.
3. Commit the hook (or adapter) together with the updated experiment record.
4. Open a PR describing the failure mode, the stable structure you anchored on,
   and the train (and, if you have it, held-out) delta.

By contributing you agree your work is licensed under the MIT License (`LICENSE`).
