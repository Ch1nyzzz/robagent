"""SopBenchAgent: function-calling agent for Amazon SOP-Bench, backed by our chat().

v0 baseline (no components): a minimal multi-turn function-calling loop
that mirrors the behavior of Amazon SOP-Bench's own FunctionCallingAgent —
but routes every LLM call through `agent.llm.chat`, which is locked to
the SUT model (deepseek-v4-pro on Together via the OpenAI-compatible API).

Component-runtime integration (v1): the loop dispatches Mounts on every
lifecycle point (SESSION_START / PRE_PROMPT_BUILD / PRE_LLM_TURN /
POST_LLM_RESPONSE / PRE_TOOL_USE / POST_TOOL_USE / PRE_FINAL_EMIT /
SESSION_END) through the SOP-Bench Dispatcher. With an empty workflow
yaml (no active components), the agent's behavior is byte-equivalent to
v0; with components active, they may rewrite the in-flight state per the
policy matrix.

Interface contract (from amazon_sop_bench.agents.base):
    execute(sop: str, task: dict, tools: ToolManager) -> AgentResult
"""
from __future__ import annotations

import json
import os
from typing import Any

from amazon_sop_bench.agents.base import AgentResult, BaseAgent

from agent.llm import chat
from agent.component_runtime_sopbench import (
    ComponentContext,
    Dispatcher,
    build_dispatcher,
)


def _bedrock_to_openai_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Bedrock-style toolSpec list into OpenAI function-calling format."""
    out: list[dict[str, Any]] = []
    for entry in specs:
        ts = entry.get("toolSpec", entry) if isinstance(entry, dict) else {}
        name = ts.get("name")
        if not name:
            continue
        description = ts.get("description", "") or ""
        schema_root = ts.get("inputSchema", {}) or {}
        if isinstance(schema_root, dict) and "json" in schema_root:
            parameters = schema_root["json"]
        elif isinstance(schema_root, dict):
            parameters = schema_root
        else:
            parameters = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
    return out


def _stringify_tool_result(tool_call: Any) -> str:
    """Format a ToolManager.execute_tool() return into a message-friendly string."""
    if not getattr(tool_call, "success", True):
        return json.dumps({"error": getattr(tool_call, "error", "tool execution failed")})
    result = getattr(tool_call, "result", None)
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _build_system_prompt(sop: str) -> str:
    return (
        "You are an SOP-following agent. Your job is to execute a Standard "
        "Operating Procedure (SOP) on a single task instance.\n\n"
        "How to operate:\n"
        "1. Read the SOP carefully — especially sections 5 (Main Procedure) "
        "and 6 (Output).\n"
        "2. Use the provided tools whenever the SOP says to gather data, "
        "validate inputs, or compute scores.\n"
        "3. Follow the SOP's decision logic step by step.\n"
        "4. When you have your final answer, output it EXACTLY in the XML "
        "format that the SOP's Output section specifies (e.g. "
        "<hazard_class>Hazard Class B</hazard_class>).\n"
        "5. Stop calling tools once you have produced the final XML output.\n"
        "6. The final XML tag is mandatory — without it the answer is "
        "considered missing.\n\n"
        "SOP DOCUMENT:\n---\n" + sop + "\n---"
    )


def _build_user_prompt(task: dict[str, Any]) -> str:
    body = "\n".join(f"{k}: {v}" for k, v in task.items())
    return (
        "Execute the SOP on this task input:\n\n"
        + body
        + "\n\nDecide which tools to call (if any), follow the SOP, and "
        "produce the final XML output."
    )


def _drain_post_llm_inject(ctx: ComponentContext) -> str:
    """Flush queued post-LLM injections into a single system reminder text."""
    queue = ctx.shared.get("post_llm_inject")
    if not queue:
        return ""
    out = "\n\n".join(s for s in queue if s and str(s).strip())
    ctx.shared["post_llm_inject"] = []
    return out


class SopBenchAgent(BaseAgent):
    """Function-calling SOP agent backed by the locked target model.

    Args:
        max_iterations: maximum assistant/tool turns before the loop bails out.
        max_tokens: per-call completion budget for chat().
        verbose_trace: when True, include the assistant's textual content in
            the reasoning trace at every turn. The OutputParser scans the
            reasoning trace for multi-field output formats, so this is useful
            even when the final answer is short.
        benchmark_name: passed into the ComponentContext for matchers that
            want to gate on domain. If unset, taken from env var
            SOPBENCH_BENCHMARK_NAME, else "unknown".
        dispatcher: optional pre-built Dispatcher. Most callers leave this
            None and let the agent fetch the cached one based on env vars
            (SOPBENCH_COMPONENT_WORKFLOW + SOPBENCH_COMPONENT_NAMES).
    """

    def __init__(
        self,
        max_iterations: int = 15,
        max_tokens: int = 8192,
        verbose_trace: bool = True,
        benchmark_name: str | None = None,
        dispatcher: Dispatcher | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.verbose_trace = verbose_trace
        self._benchmark_name = benchmark_name
        self._dispatcher_override = dispatcher

    # --- runtime hook -----------------------------------------------------

    def _dispatcher(self) -> Dispatcher:
        if self._dispatcher_override is not None:
            return self._dispatcher_override
        return build_dispatcher()

    def _benchmark(self) -> str:
        return (
            self._benchmark_name
            or os.environ.get("SOPBENCH_BENCHMARK_NAME", "unknown")
        )

    # --- main loop --------------------------------------------------------

    def execute(
        self,
        sop: str,
        task: dict[str, Any],
        tools: Any,
    ) -> AgentResult:
        trace_lines: list[str] = []
        executed_tool_calls: list[dict[str, Any]] = []
        try:
            tool_specs = tools.get_tool_specs() if hasattr(tools, "get_tool_specs") else []
            oai_tools = _bedrock_to_openai_tools(tool_specs)
            dispatcher = self._dispatcher()
            ctx = ComponentContext(
                benchmark=self._benchmark(),
                task_id=str(task.get("__task_id__", "")) if isinstance(task, dict) else "",
                sop_text=sop,
                task_input=dict(task) if isinstance(task, dict) else {},
                tool_specs=tool_specs,
                system_prompt=_build_system_prompt(sop),
                user_prompt=_build_user_prompt(task),
            )
            ctx.executed_tool_calls = executed_tool_calls
            # Phase C: wire ctx.chat / ctx.emit per the locked SUT model.
            dispatcher.wire_capabilities(ctx)

            # Tier-1 task_received: lifecycle anchor right after ctx construction.
            dispatcher.emit("task_received", ctx)
            if ctx.blocked:
                return self._blocked_result(ctx, trace_lines, executed_tool_calls)

            # SESSION_START: static system_prompt injection only.
            dispatcher.emit("session_start", ctx)
            if ctx.blocked:
                return self._blocked_result(ctx, trace_lines, executed_tool_calls)

            # PRE_PROMPT_BUILD: can rewrite user_prompt or inject into system_prompt.
            dispatcher.emit("pre_prompt_build", ctx)
            if ctx.blocked:
                return self._blocked_result(ctx, trace_lines, executed_tool_calls)
            # Tier-1 alias for cross-sibling consistency.
            dispatcher.emit("pre_context_build", ctx)
            if ctx.blocked:
                return self._blocked_result(ctx, trace_lines, executed_tool_calls)

            # Tier-1 pre_agent_construct: last chance to influence the inference
            # request shape before messages list is sealed.
            dispatcher.emit("pre_agent_construct", ctx)
            if ctx.blocked:
                return self._blocked_result(ctx, trace_lines, executed_tool_calls)

            ctx.messages = [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user", "content": ctx.user_prompt},
            ]
            final_content: str = ""

            for turn in range(1, self.max_iterations + 1):
                ctx.turn_index = turn

                # Drain any post-LLM injections from the previous turn into a
                # synthetic system reminder before the next chat call.
                reminder = _drain_post_llm_inject(ctx)
                if reminder:
                    ctx.messages.append({"role": "system", "content": reminder})
                    trace_lines.append(
                        f"[turn {turn}] injected reminder ({len(reminder)} chars)"
                    )

                # PRE_LLM_TURN: components can rewrite messages list.
                dispatcher.emit("pre_llm_turn", ctx)
                if ctx.blocked:
                    return self._blocked_result(ctx, trace_lines, executed_tool_calls)

                # Tier-1 pre_llm_request: anything that wants to react just
                # before the SUT call (sub-LLM verifier prep, last-pass
                # injection without rewriting `ctx.messages`).
                dispatcher.emit("pre_llm_request", ctx)
                if ctx.blocked:
                    return self._blocked_result(ctx, trace_lines, executed_tool_calls)

                resp = chat(
                    ctx.messages,
                    tools=oai_tools if oai_tools else None,
                    max_tokens=self.max_tokens,
                )
                content = resp.get("content") or ""
                finish_reason = resp.get("finish_reason") or ""
                tcs = list(resp.get("tool_calls") or [])
                trace_lines.append(f"[turn {turn}] finish_reason={finish_reason}")
                if self.verbose_trace and content:
                    trace_lines.append(content)

                # Round-trip the assistant message so subsequent turns see it.
                ctx.messages.append(resp["assistant_message"])

                # Update ctx for POST_LLM_RESPONSE.
                ctx.raw_response = content
                ctx.finish_reason = finish_reason
                ctx.tool_calls = tcs
                dispatcher.emit("post_llm_response", ctx)
                if ctx.blocked:
                    return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                # Tier-1 post_llm_response_raw + synthesised failure-mode events.
                dispatcher.emit("post_llm_response_raw", ctx)
                if ctx.blocked:
                    return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                if finish_reason == "length":
                    dispatcher.emit("on_length_truncation", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                if not (ctx.raw_response or "").strip():
                    dispatcher.emit("on_empty_response", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                if not tcs:
                    dispatcher.emit("on_no_tool_call_emitted", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)

                # If a component REWROTE raw_response and there are no tool
                # calls, treat the rewritten content as final.
                if not tcs:
                    final_content = ctx.raw_response
                    break

                # Per-tool-call dispatch.
                for tc in tcs:
                    name = tc.get("name") or ""
                    args = tc.get("arguments")
                    if not isinstance(args, dict):
                        args = {}
                    ctx.current_tool_name = name
                    ctx.current_tool_args = dict(args)
                    ctx.current_tool_call_id = tc.get("id") or ""
                    ctx.current_tool_result = None
                    ctx.current_tool_result_str = ""
                    ctx.current_tool_success = True
                    ctx.current_tool_error = None

                    # Tier-1 pre_tool_arg_validation: narrow phase for
                    # deterministic schema / cross-arg consistency checks
                    # before the main PRE_TOOL_USE class×event matrix runs.
                    dispatcher.emit("pre_tool_arg_validation", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                    if ctx.shared.get("skip_current_tool"):
                        # An arg-validator component decided to skip; fall
                        # through to the existing skip-handling branch below
                        # without re-emitting PRE_TOOL_USE.
                        pass

                    dispatcher.emit("pre_tool_use", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)

                    if ctx.shared.get("skip_current_tool"):
                        reason = ctx.shared.pop("skip_current_tool_reason",
                                                 f"blocked by component")
                        result_str = json.dumps({"blocked_by_component": reason})
                        ctx.current_tool_result_str = result_str
                        ctx.current_tool_success = False
                        ctx.current_tool_error = reason
                        trace_lines.append(
                            f"[tool] {name} skipped by component: {reason}"
                        )
                        ctx.shared.pop("skip_current_tool", None)
                    else:
                        if not hasattr(tools, "execute_tool"):
                            trace_lines.append(
                                f"[tool] no execute_tool on tools instance; skipping {name}"
                            )
                            continue
                        tool_call_result = tools.execute_tool(
                            name, ctx.current_tool_args
                        )
                        result_str = _stringify_tool_result(tool_call_result)
                        ctx.current_tool_result = getattr(tool_call_result, "result", None)
                        ctx.current_tool_result_str = result_str
                        ctx.current_tool_success = bool(
                            getattr(tool_call_result, "success", True)
                        )
                        ctx.current_tool_error = getattr(tool_call_result, "error", None)
                        trace_lines.append(
                            f"[tool] {name}({json.dumps(ctx.current_tool_args, default=str)}) -> "
                            f"success={ctx.current_tool_success}"
                        )

                    dispatcher.emit("post_tool_use", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                    # Tier-1 post_tool_result_raw + on_tool_error.
                    dispatcher.emit("post_tool_result_raw", ctx)
                    if ctx.blocked:
                        return self._blocked_result(ctx, trace_lines, executed_tool_calls)
                    if not ctx.current_tool_success:
                        dispatcher.emit("on_tool_error", ctx)
                        if ctx.blocked:
                            return self._blocked_result(ctx, trace_lines, executed_tool_calls)

                    executed_tool_calls.append({
                        "tool": name,
                        "parameters": ctx.current_tool_args,
                        "result": ctx.current_tool_result,
                        "success": ctx.current_tool_success,
                        "error": ctx.current_tool_error,
                    })
                    ctx.messages.append({
                        "role": "tool",
                        "tool_call_id": ctx.current_tool_call_id,
                        "content": ctx.current_tool_result_str,
                    })
            else:
                # Loop exhausted without a tool-free assistant response.
                final_content = ctx.raw_response or "(max_iterations exhausted without final output)"
                trace_lines.append("[warn] max_iterations exhausted")
                # Tier-1 on_explicit_terminate: an exit-gate component may
                # block emission entirely (e.g. require an XML answer tag).
                dispatcher.emit("on_explicit_terminate", ctx)
                if ctx.blocked:
                    return self._blocked_result(ctx, trace_lines, executed_tool_calls)

            ctx.final_output = final_content
            dispatcher.emit("pre_final_emit", ctx)
            if ctx.blocked:
                return self._blocked_result(ctx, trace_lines, executed_tool_calls)
            dispatcher.emit("session_end", ctx)

            return AgentResult(
                output=ctx.final_output,
                tool_calls=executed_tool_calls,
                reasoning_trace="\n".join(trace_lines),
                success=True,
            )
        except BaseException as exc:  # noqa: BLE001 - surface any failure as agent error
            return AgentResult(
                output="",
                tool_calls=executed_tool_calls,
                reasoning_trace="\n".join(trace_lines),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _blocked_result(
        ctx: ComponentContext,
        trace_lines: list[str],
        executed_tool_calls: list[dict[str, Any]],
    ) -> AgentResult:
        trace_lines.append(f"[blocked] {ctx.blocked_reason}")
        return AgentResult(
            output=ctx.final_output or "",
            tool_calls=executed_tool_calls,
            reasoning_trace="\n".join(trace_lines),
            success=False,
            error=f"blocked: {ctx.blocked_reason}",
        )
