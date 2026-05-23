"""Hook runtime for tau2 candidates.

A candidate produced by the `hook-harness-tau2` skill is NOT a new
`LLMAgent` subclass — it is a single hook file under `agent_tau2/hooks/`.
At evaluation time the runtime composes the frontier hook set ∪ {candidate}
into a `HookedLLMAgent` wrapping the stock `LLMAgent`.

Module layout
-------------
    types.py     — Hook, HookEvent, HookClass, Decision, HookContext
    policy.py    — class × event × decision-kind permission matrix
    registry.py  — load hooks from a directory / file list
    agent.py     — HookedLLMAgent + build_agent(...) entry point used by
                   tau2_runner.py (no runner change required).
"""
