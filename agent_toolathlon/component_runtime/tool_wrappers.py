"""v2: wrap each connected MCP tool as an SDK FunctionTool so the
component runtime can intercept BEFORE and AFTER the real invocation
with the actual arguments and the actual result string.

Why this exists
---------------
The OpenAI Agents SDK exposes `AgentHooks.on_tool_start(ctx, agent, tool)`
which does NOT carry the LLM-emitted `arguments` dict. That means in v1,
PRE_TOOL_USE components could only match by tool NAME — REWRITE_TOOL_ARGS
was unimplementable, and BLOCK could only be advisory (the SDK was already
invoking the tool when the hook fired).

By replacing MCP servers wired into `Agent(mcp_servers=[...])` with
FunctionTools wired into `Agent(tools=[...])`, we control the entire
invocation: parse args → PRE_TOOL_USE dispatch (real args) → maybe
rewrite/block → call MCP → POST_TOOL_USE dispatch → maybe concatenate
inject into result string (LLM sees it on its next inference).

Compatibility
-------------
We REUSE Toolathlon's monkey-patched `MCPUtil.invoke_mcp_tool` for the
real invocation step — it carries the overlong-output truncation logic
and the `gw-<server>-<tool>` naming. So the only behavioural delta vs.
v1 (no wrap) is: components can now actually REWRITE/BLOCK on the args
the LLM emitted, and POST_TOOL_USE INJECT_CONTEXT lands in the same
turn instead of being queued.
"""
from __future__ import annotations

import json
from typing import Any, Optional

# The monkey patch at Toolathlon-src/utils/openai_agents_monkey_patch/custom_mcp_util.py
# replaces MCPUtil.to_function_tool and MCPUtil.invoke_mcp_tool to (1)
# prefix tool names with "<server.name>-" and (2) handle overlong output
# truncation. We import the patched class so wrap names match v0 / v1
# behaviour and overlong handling is preserved.
from agents.mcp.util import MCPUtil  # type: ignore[import-untyped]
from agents.tool import FunctionTool  # type: ignore[import-untyped]
from agents.strict_schema import ensure_strict_json_schema  # type: ignore[import-untyped]

from .hooks import (
    ComponentDispatcher,
    _shared_from_ctx,
    _queue_post_tool_use_text,
)
from .types import Decision, DecisionKind


def _wrapped_tool_full_name(server_name: str, tool_name: str) -> str:
    """Mirror the name format produced by Toolathlon's monkey-patched
    MCPUtil.to_function_tool (custom_mcp_util.py:36-38)."""
    return f"{server_name}-{tool_name}"


def _strictify_schema(schema: dict) -> tuple[dict, bool]:
    if "properties" not in schema:
        schema = {**schema, "properties": {}}
    try:
        return ensure_strict_json_schema(schema), True
    except Exception:
        return schema, False


def _component_block_message(component_name: str, full_tool_name: str,
                             reason: str) -> str:
    return (
        f"<component_block component={component_name} tool={full_tool_name}>\n"
        f"Tool invocation was blocked by component runtime.\n"
        f"Reason: {reason or '(no reason given)'}\n"
        f"Do not retry this call with the same shape."
        f"\n</component_block>"
    )


class ComponentMCPToolWrapper:
    """Callable bound to a single (server, mcp_tool) pair.

    Instance attributes are stable across calls so it's safe to put one
    of these into a FunctionTool's `on_invoke_tool` slot. The wrapper
    delegates the actual MCP call to upstream MCPUtil.invoke_mcp_tool to
    keep truncation + overlong-output handling identical to v1.
    """

    def __init__(self, server, mcp_tool, dispatcher: ComponentDispatcher):
        self.server = server
        self.mcp_tool = mcp_tool
        self.dispatcher = dispatcher
        self.full_name = _wrapped_tool_full_name(server.name, mcp_tool.name)

    async def __call__(self, context, input_json: str) -> str:
        # 1. parse args
        try:
            args: dict[str, Any] = json.loads(input_json) if input_json else {}
        except Exception:
            # Match upstream behaviour: let SDK Runner surface the error.
            return await MCPUtil.invoke_mcp_tool(
                self.server, self.mcp_tool, context, input_json
            )
        if not isinstance(args, dict):
            args = {}

        shared = _shared_from_ctx(context)

        # 2. PRE_TOOL_USE dispatch (REAL args)
        pre_decision = self.dispatcher.fire_pre_tool_use_with_args(
            shared, self.full_name, args,
        )
        if pre_decision.kind is DecisionKind.BLOCK:
            # True BLOCK: do NOT call the tool. Return a recognisable
            # component-block string as the tool's output.
            return _component_block_message(
                component_name=pre_decision.reason or "<unknown_component>",
                full_tool_name=self.full_name,
                reason=pre_decision.reason or "",
            )
        if pre_decision.kind is DecisionKind.REWRITE_TOOL_ARGS:
            args = dict(pre_decision.payload)

        # 3. real MCP invocation through upstream (preserves truncation +
        # overlong-output handling)
        new_input_json = json.dumps(args, ensure_ascii=False)
        result_str = await MCPUtil.invoke_mcp_tool(
            self.server, self.mcp_tool, context, new_input_json,
        )

        # 4. POST_TOOL_USE dispatch (REAL args + REAL result)
        post_decision = self.dispatcher.fire_post_tool_use_inline(
            shared, self.full_name, args, result_str,
        )
        if post_decision.kind is DecisionKind.INJECT_CONTEXT:
            inject_text = str(post_decision.payload)
            result_str = (
                f"{result_str}\n\n"
                f"<components_post_tool_use>\n{inject_text}\n"
                f"</components_post_tool_use>"
            )

        return result_str


async def wrap_mcp_tools_as_function_tools(
    mcp_manager,
    dispatcher: ComponentDispatcher,
    *,
    convert_schemas_to_strict: bool = False,
) -> list[FunctionTool]:
    """Enumerate every tool on every connected MCP server and produce a
    list of SDK FunctionTools bound to ComponentMCPToolWrapper.

    Each FunctionTool is tagged with `_cr_wrapped = True` so
    ComponentAgentHooks knows to skip its own dispatch (the wrapper
    already did it with real args).
    """
    wrapped: list[FunctionTool] = []
    seen_names: set[str] = set()
    for server in mcp_manager.get_all_connected_servers():
        mcp_tools = await server.list_tools()
        for mcp_tool in mcp_tools:
            full_name = _wrapped_tool_full_name(server.name, mcp_tool.name)
            if full_name in seen_names:
                # mirror upstream MCPUtil.get_all_function_tools duplicate
                # behaviour (it raises, but we soft-skip here to avoid
                # breaking a multi-server run on a name clash; the
                # earliest server wins).
                continue
            seen_names.add(full_name)

            schema = dict(mcp_tool.inputSchema or {})
            schema, is_strict = (
                _strictify_schema(schema) if convert_schemas_to_strict
                else (
                    {**schema, "properties": schema.get("properties", {})},
                    False,
                )
            )

            ftool = FunctionTool(
                name=full_name,
                description=mcp_tool.description or "",
                params_json_schema=schema,
                on_invoke_tool=ComponentMCPToolWrapper(server, mcp_tool, dispatcher),
                strict_json_schema=is_strict,
            )
            # Marker the hooks check for skip-double-dispatch.
            try:
                object.__setattr__(ftool, "_cr_wrapped", True)
            except Exception:
                # FunctionTool is a frozen pydantic / dataclass in some
                # SDK versions; fall back to attaching via dict.
                ftool.__dict__["_cr_wrapped"] = True
            wrapped.append(ftool)
    return wrapped
