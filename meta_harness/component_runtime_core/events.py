"""Tier-1 event name constants (Phase B of event-runtime migration).

Runtime emits these at the lifecycle anchors only it can see. Components
subscribe via `Component(listens="<name>", ...)`. The constants are
strings — the runtime does NOT enforce that a sibling emits or that a
component listens to a specific Tier-1 name; they exist for
autocomplete, type hints, and a single place to look up the canonical
spelling.

Tier 1 = runtime-emitted, common across siblings where applicable.
Tier 2/3 = component-emitted custom events; open string namespace, see
SKILL.md for naming convention (`iter<N>_<slug>_<event>` for evolution,
`on_<thing>` for failure-mode events).

Per-sibling applicability is documented in the plan file's Layer-5
matrix; not every sibling emits every Tier-1 event (gaia has no tool
loop, toolathlon has SDK-internal events we cannot intercept without
ModelProvider wrapping, etc.).
"""

# Setup phase
TASK_RECEIVED       = "task_received"
PRE_CONTEXT_BUILD   = "pre_context_build"
PRE_AGENT_CONSTRUCT = "pre_agent_construct"

# Per-LLM-turn
PRE_LLM_REQUEST         = "pre_llm_request"
POST_LLM_RESPONSE_RAW   = "post_llm_response_raw"
ON_LENGTH_TRUNCATION    = "on_length_truncation"
ON_EMPTY_RESPONSE       = "on_empty_response"
ON_NO_TOOL_CALL_EMITTED = "on_no_tool_call_emitted"

# Per-tool-call
PRE_TOOL_ARG_VALIDATION = "pre_tool_arg_validation"
PRE_TOOL_USE            = "pre_tool_use"
POST_TOOL_RESULT_RAW    = "post_tool_result_raw"
ON_TOOL_ERROR           = "on_tool_error"
POST_TOOL_USE           = "post_tool_use"

# Termination
ON_EXPLICIT_TERMINATE = "on_explicit_terminate"
SESSION_END           = "session_end"

# Convenience tuple — useful for skill table rendering / smoke tests
TIER_1: tuple[str, ...] = (
    TASK_RECEIVED,
    PRE_CONTEXT_BUILD,
    PRE_AGENT_CONSTRUCT,
    PRE_LLM_REQUEST,
    POST_LLM_RESPONSE_RAW,
    ON_LENGTH_TRUNCATION,
    ON_EMPTY_RESPONSE,
    ON_NO_TOOL_CALL_EMITTED,
    PRE_TOOL_ARG_VALIDATION,
    PRE_TOOL_USE,
    POST_TOOL_RESULT_RAW,
    ON_TOOL_ERROR,
    POST_TOOL_USE,
    ON_EXPLICIT_TERMINATE,
    SESSION_END,
)
