"""tau2 candidate v0 — the stock LLMAgent, unmodified.

A tau2 candidate is a module exposing:

    build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent

`tools` and `domain_policy` come from the tau2 environment; `kwargs` carries
`llm` (model name) and `llm_args` (provider args). The returned object must be
a tau2 `HalfDuplexAgent` — `LLMAgent` is the reference implementation and the
baseline every evolved candidate starts from.

An evolved candidate keeps this signature but returns a subclass of `LLMAgent`
(or a wrapper) that pushes work into deterministic code: validating tool-call
arguments before they are emitted, enforcing domain-policy preconditions,
normalising the final message, etc. The LLM stays in the loop only for the
genuine turn-by-turn decisions.
"""
from __future__ import annotations

from tau2.agent.llm_agent import LLMAgent


def build_agent(tools, domain_policy, **kwargs):
    return LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
