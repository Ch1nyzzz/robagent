"""Toolathlon candidate cr — component-runtime-enabled TaskAgent.

`build_agent` re-exported from component_runtime.agent. host_agent_loop
resolves `agent_toolathlon.<candidate>.agent::build_agent` via the
TOOLATHLON_CANDIDATE env var, so `TOOLATHLON_CANDIDATE=cr` swaps the
baseline v0 PrettyDecoupledTaskAgent for the CR-aware variant.

Component selection / workflow resolution / hook wiring all live in
`agent_toolathlon.component_runtime.agent.build_agent`; this file is a
thin re-export so the candidate folder layout matches v0's.
"""
from __future__ import annotations

from agent_toolathlon.component_runtime.agent import build_agent

__all__ = ["build_agent"]
