from __future__ import annotations

from agent.component_runtime.types import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, Mount, StateScope, Trust,
)
from agent.llm import chat

_FILE_CONTEXT_MARKER = "[Attached file content:"

_MINIMAL_RECOVERY_SYSTEM = (
    "You are answering a benchmark task. "
    "Output ONLY the final answer — a number or short phrase. "
    "No chain-of-thought, no reasoning, no explanation."
)


def _matches(ctx: ComponentContext) -> bool:
    return ctx.shared.get("finish_reason") == "length" and not ctx.raw_response


def _handler(ctx: ComponentContext) -> Decision:
    # When gaia_file_channel has injected file content into ctx.system_prompt
    # (detectable by the stable "[Attached file content:" prefix), the recovery
    # call must use that full system_prompt — otherwise the model has no file
    # data and refuses or hallucinates. For non-file tasks, use the minimal
    # hardcoded prompt to preserve existing recovery behaviour exactly.
    has_file_context = _FILE_CONTEXT_MARKER in (ctx.system_prompt or "")
    system_content = ctx.system_prompt if has_file_context else _MINIMAL_RECOVERY_SYSTEM

    recovery_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": ctx.prompt or ""},
    ]
    for budget in (8192, 32768):
        try:
            result = chat(messages=recovery_messages, max_tokens=budget)
            content = (result.get("content") or "").strip()
            if content:
                return Decision.rewrite(content)
        except Exception:
            pass
    return Decision.allow()


COMPONENT = Component(
    name="length_recovery_guard",
    cls=ComponentClass.REACTIVE_GUARD,
    mount=Mount.POST_LLM_RESPONSE,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.LLM_CALL,),
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "LLM API fields 'finish_reason' (value 'length') and empty 'content' string "
            "— stable OpenAI-compatible completion API contract. "
            "ComponentContext.system_prompt at POST_LLM_RESPONSE contains all "
            "INJECT_CONTEXT payloads accumulated at SESSION_START and PRE_PROMPT_BUILD "
            "— stable component_runtime/base.py dispatch contract. "
            "'[Attached file content:' is gaia_file_channel's stable injection prefix "
            "(component_iter4_gaia_file_channel.py), independent of task content."
        ),
        blast_radius="local",
        rollback_when=(
            "train-30 accuracy decreases vs iter3 frontier on tasks not in the "
            "length-truncation + file-context failure set, indicating the selective "
            "file-context recovery regresses previously correct recovery results"
        ),
        out_of_evidence_probe=(
            "Task where finish_reason='length', raw_response is empty, and "
            "ctx.system_prompt does NOT contain '[Attached file content:' (no file "
            "injected): handler uses hardcoded _MINIMAL_RECOVERY_SYSTEM — identical to "
            "iter3 behaviour, no change in outcome. "
            "Task where finish_reason='stop' with non-empty content: matcher returns "
            "False, handler never fires."
        ),
        fallback=(
            "matcher returns False when finish_reason != 'length' or raw_response is "
            "non-empty; handler uses hardcoded minimal prompt when no file context "
            "present; handler returns allow() if all recovery attempts yield empty content"
        ),
    ),
)
