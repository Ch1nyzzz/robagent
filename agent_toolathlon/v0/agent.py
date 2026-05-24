"""Toolathlon candidate v0 — the stock TaskAgent, unmodified.

A Toolathlon candidate exposes:

    build_agent(*, task_config, agent_config, agent_model_provider,
                user_config, user_client, mcp_config,
                termination_checker, allow_resume, single_turn_mode,
                **kwargs) -> TaskAgent

`task_config`, `agent_config`, etc. come from the Toolathlon runtime (see
`scripts/decoupled/host_agent_loop.py::run_host_loop` for the canonical
wiring). The returned object must be (or subclass) `utils.roles.task_agent.
TaskAgent` — the reference half-duplex agent driving the OpenAI Agents SDK
loop over MCP tools.

An evolved candidate keeps this signature but returns a TaskAgent subclass
that pushes work into deterministic code, typically by attaching
`agent_hooks` / `run_hooks` (OpenAI Agents SDK lifecycle hooks) that
component_runtime fires at PRE_TOOL_USE / POST_TOOL_USE / etc. The LLM
stays in the loop only for genuine turn-by-turn decisions.
"""
from __future__ import annotations

from typing import Any

# This import only resolves with Toolathlon-src on sys.path. The runner
# (toolathlon_runner.py) is responsible for adding it before importing
# this module.
from utils.roles.task_agent import TaskAgent  # type: ignore


def build_agent(
    *,
    task_config: Any,
    agent_config: Any,
    agent_model_provider: Any,
    user_config: Any,
    user_client: Any,
    mcp_config: Any,
    termination_checker: Any = None,
    allow_resume: bool = False,
    single_turn_mode: bool = False,
    agent_hooks: Any = None,
    run_hooks: Any = None,
    debug: bool = False,
    manual: bool = False,
    pretty_base: Any = None,
    **_unused: Any,
) -> TaskAgent:
    cls = pretty_base if pretty_base is not None else TaskAgent
    return cls(
        task_config=task_config,
        agent_config=agent_config,
        agent_model_provider=agent_model_provider,
        user_config=user_config,
        user_client=user_client,
        mcp_config=mcp_config,
        agent_hooks=agent_hooks,
        run_hooks=run_hooks,
        termination_checker=termination_checker,
        debug=debug,
        allow_resume=allow_resume,
        manual=manual,
        single_turn_mode=single_turn_mode,
    )
