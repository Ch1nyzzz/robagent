"""EnterpriseOpsAgent: component-enabled wrapper around upstream EnterpriseOps-Gym.

Strategy
--------
Reuse upstream `third_party/EnterpriseOps-Gym/benchmark/executor.py` +
`orchestrators/react.py` (which already handle MCP server lifecycle, DB
seeding, tool dispatch routing across multiple gym servers, and SQL
verifier execution). Inject `Dispatcher.dispatch(mount, ctx)` calls at the
8 component-runtime mount points:

  SESSION_START / PRE_PROMPT_BUILD  — before the agent loop
  PRE_LLM_TURN / POST_LLM_RESPONSE  — per turn inside our ReactOrchestrator subclass
  PRE_TOOL_USE / POST_TOOL_USE      — per MCP tool call inside the same subclass
  PRE_FINAL_EMIT / SESSION_END      — after the loop, after upstream's verifiers

v0 baseline parity: with an empty workflow yaml, dispatcher hooks become
no-ops and behavior is byte-equivalent to vanilla upstream.

Interface
---------
    asyncio.run(run_task(task_config_dict, llm_config, ...)) -> dict
    The returned dict is the upstream `BenchmarkExecutor.execute_benchmark()`
    result (with `runs`, `statistics`, `benchmark_config`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make upstream imports resolvable.
_ROOT = Path(__file__).resolve().parent.parent
_UPSTREAM = _ROOT / "third_party" / "EnterpriseOps-Gym"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from langchain_core.messages import (  # noqa: E402
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# upstream
from benchmark.executor import BenchmarkExecutor  # noqa: E402
from benchmark.models import BenchmarkConfig, LLMConfig  # noqa: E402
from orchestrators.react import ReactOrchestrator  # noqa: E402

from agent.component_runtime_enterpriseops import (  # noqa: E402
    ComponentContext,
    Dispatcher,
    build_dispatcher,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReactOrchestrator subclass with dispatcher hooks
# ---------------------------------------------------------------------------


class EnterpriseOpsReactOrchestrator(ReactOrchestrator):
    """Upstream ReactOrchestrator + per-turn / per-tool dispatcher hooks.

    Keyword-only extras (popped before passing to base):
      dispatcher : Dispatcher | None
      ctx        : ComponentContext | None

    With either left None, falls back to upstream's stock execute(); this
    keeps the subclass safe to install as the default orchestrator class
    even when the component runtime hasn't been initialised.
    """

    def __init__(self, *args: Any, dispatcher: Optional[Dispatcher] = None,
                 ctx: Optional[ComponentContext] = None,
                 **kwargs: Any) -> None:
        self._dispatcher = dispatcher
        self._ctx = ctx
        super().__init__(*args, **kwargs)

    async def execute(self) -> Dict[str, Any]:
        disp, ctx = self._dispatcher, self._ctx
        if disp is None or ctx is None:
            return await super().execute()

        # SESSION_START + PRE_PROMPT_BUILD have already fired in run_task
        # before BenchmarkConfig was built; ctx.system_prompt / ctx.user_prompt
        # may carry injections from those mounts.
        sys_prompt = ctx.system_prompt or self.config.system_prompt
        usr_prompt = ctx.user_prompt or self.config.user_prompt

        messages: List[BaseMessage] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        conversation_flow: List[dict] = [
            {"type": "system_message", "content": sys_prompt},
            {"type": "user_message", "content": usr_prompt},
        ]
        tools_used: List[str] = []
        tool_results: List[dict] = []

        # Snapshot tool_specs into ctx so matchers can read them.
        ctx.tool_specs = list(self.available_tools)

        for iteration in range(self.max_iterations):
            ctx.turn_index = iteration + 1

            # Drain any pending POST_LLM_RESPONSE injections from previous turn.
            queue = ctx.shared.pop("post_llm_inject", None)
            if queue:
                joined = "\n\n".join(s for s in queue if s and str(s).strip())
                if joined:
                    messages.append(SystemMessage(content=joined))
                    conversation_flow.append({
                        "type": "system_reminder",
                        "content": joined,
                    })

            # PRE_LLM_TURN -------------------------------------------------
            ctx.messages = list(messages)
            disp.emit("pre_llm_turn", ctx)
            if ctx.blocked:
                break
            # Component may have REWRITTEN messages.
            if ctx.messages and ctx.messages is not messages:
                messages = list(ctx.messages)

            # Tier-1 pre_llm_request: react just before the SUT call.
            disp.emit("pre_llm_request", ctx)
            if ctx.blocked:
                break

            # LLM call -----------------------------------------------------
            response = await self.llm_client.invoke_with_tools(
                messages, self.available_tools
            )
            messages.append(response)

            usage_metadata = getattr(response, "usage_metadata", {}) or {}
            response_metadata = getattr(response, "response_metadata", {}) or {}
            response_tool_calls = response.tool_calls or []

            conversation_flow.append({
                "type": "ai_message",
                "content": response.content,
                "usage_metadata": usage_metadata,
                "response_metadata": response_metadata,
                "tool_calls": [
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in response_tool_calls
                ],
            })

            # POST_LLM_RESPONSE -------------------------------------------
            ctx.raw_response = response.content or ""
            ctx.tool_calls = [
                {"name": tc["name"], "arguments": tc.get("args", {}),
                 "id": tc.get("id", "")}
                for tc in response_tool_calls
            ]
            ctx.finish_reason = str(response_metadata.get("finish_reason", ""))
            disp.emit("post_llm_response", ctx)
            if ctx.blocked:
                break
            # Tier-1 post_llm_response_raw + synthesised failure-mode events.
            disp.emit("post_llm_response_raw", ctx)
            if ctx.blocked:
                break
            if ctx.finish_reason == "length":
                disp.emit("on_length_truncation", ctx)
                if ctx.blocked:
                    break
            if not (ctx.raw_response or "").strip():
                disp.emit("on_empty_response", ctx)
                if ctx.blocked:
                    break
            if not response_tool_calls:
                disp.emit("on_no_tool_call_emitted", ctx)
                if ctx.blocked:
                    break

            # No tool calls → done (parity with upstream).
            if not response_tool_calls:
                # Tier-1 on_explicit_terminate: a gate component may BLOCK
                # the final emission (e.g. require a verifier-friendly tag).
                disp.emit("on_explicit_terminate", ctx)
                if ctx.blocked:
                    break
                break

            # Per-tool-call loop -------------------------------------------
            for tc in response_tool_calls:
                tool_name = tc["name"]
                raw_args = tc.get("args") or {}
                tool_args = dict(raw_args) if isinstance(raw_args, dict) else {}

                ctx.current_tool_name = tool_name
                ctx.current_tool_args = tool_args
                ctx.current_tool_call_id = tc.get("id", "") or ""
                ctx.current_tool_result = None
                ctx.current_tool_result_str = ""
                ctx.current_tool_success = True
                ctx.current_tool_error = None
                ctx.current_tool_server = self.tool_to_server_mapping.get(
                    tool_name, ""
                )

                # Tier-1 pre_tool_arg_validation: narrow phase for
                # deterministic schema / cross-arg consistency checks
                # before the main PRE_TOOL_USE class×event matrix fires.
                disp.emit("pre_tool_arg_validation", ctx)
                if ctx.blocked:
                    break

                # PRE_TOOL_USE ---------------------------------------------
                disp.emit("pre_tool_use", ctx)
                if ctx.blocked:
                    break

                if ctx.shared.get("skip_current_tool"):
                    reason = ctx.shared.pop(
                        "skip_current_tool_reason", "blocked by component"
                    )
                    skipped_payload = {"blocked_by_component": reason}
                    tool_result_outer = {
                        "result": {"success": False, "result": skipped_payload,
                                   "error": reason},
                        "gym_server": ctx.current_tool_server,
                    }
                    ctx.current_tool_result = skipped_payload
                    ctx.current_tool_result_str = json.dumps(skipped_payload)
                    ctx.current_tool_success = False
                    ctx.current_tool_error = reason
                    ctx.shared.pop("skip_current_tool", None)
                else:
                    exec_result = await self._execute_tool_call(
                        tool_name, ctx.current_tool_args
                    )
                    tool_result_outer = exec_result
                    inner = exec_result.get("result", {}) or {}
                    inner_payload = inner.get("result", {})
                    ctx.current_tool_result = inner_payload
                    ctx.current_tool_result_str = json.dumps(
                        inner_payload, default=str
                    )
                    ctx.current_tool_success = bool(inner.get("success", True))
                    ctx.current_tool_error = inner.get("error")

                if tool_name not in tools_used:
                    tools_used.append(tool_name)
                tool_results.append({
                    "tool_name": tool_name,
                    "arguments": ctx.current_tool_args,
                    "result": tool_result_outer.get("result"),
                    "gym_server": tool_result_outer.get(
                        "gym_server", ctx.current_tool_server
                    ),
                })

                # POST_TOOL_USE --------------------------------------------
                disp.emit("post_tool_use", ctx)
                if ctx.blocked:
                    break
                # Tier-1 post_tool_result_raw + on_tool_error.
                disp.emit("post_tool_result_raw", ctx)
                if ctx.blocked:
                    break
                if not ctx.current_tool_success:
                    disp.emit("on_tool_error", ctx)
                    if ctx.blocked:
                        break

                # Append the (possibly rewritten) tool result back to the messages.
                messages.append(ToolMessage(
                    content=ctx.current_tool_result_str,
                    tool_call_id=ctx.current_tool_call_id,
                ))
                conversation_flow.append({
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "result": ctx.current_tool_result,
                    "gym_server": ctx.current_tool_server,
                })
                ctx.executed_tool_calls.append({
                    "tool": tool_name,
                    "parameters": ctx.current_tool_args,
                    "result": ctx.current_tool_result,
                    "success": ctx.current_tool_success,
                    "error": ctx.current_tool_error,
                })

            if ctx.blocked:
                break

        return {
            "final_response": messages[-1].content if messages else "",
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def _resolve_seed_paths(cfg_data: dict) -> None:
    """Rewrite any relative seed_database_file under gym_servers_config to
    an absolute path rooted at the upstream dir. Avoids depending on CWD.
    """
    for gym in cfg_data.get("gym_servers_config", []) or []:
        sdf = gym.get("seed_database_file")
        if sdf and not Path(sdf).is_absolute():
            gym["seed_database_file"] = str((_UPSTREAM / sdf).resolve())


async def run_task(
    task_config_dict: Dict[str, Any],
    llm_config: LLMConfig,
    *,
    domain: str = "unknown",
    task_id: str = "",
    workflow_path: Optional[Path] = None,
    components_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
    component_state_dir: Optional[Path] = None,
    config_path: str = "<inline>",
) -> Dict[str, Any]:
    """Run one EnterpriseOps-Gym task end-to-end through the component-enabled agent.

    Parameters
    ----------
    task_config_dict
        A dict in the same shape upstream's `evaluate.py` produces from HF
        rows: must contain `system_prompt`, `user_prompt`, `gym_servers_config`,
        `verifiers` (and may contain `selected_tools`, `user_info`, etc.).
    llm_config
        Upstream LLMConfig dataclass.
    domain
        Domain slug for ComponentContext.benchmark (e.g. "calendar", "itsm").
    task_id
        Task identifier for ComponentContext.task_id.
    workflow_path
        Path to the workflow YAML (e.g.
        `meta_harness/workflows/enterpriseops_calendar.yaml`). If None,
        falls back to ENTERPRISEOPS_COMPONENT_WORKFLOW env var, else empty
        workflow (= no components = v0 parity).
    components_dir
        Directory containing component .py files. If None, env var
        ENTERPRISEOPS_COMPONENT_DIR, else COMPONENTS_DIR_DEFAULT.
    run_tag
        Run identifier for the dispatcher trace sink. Defaults to "default".
    component_state_dir
        Override for the dispatcher's trace base dir.
    config_path
        Stored on the upstream BenchmarkExecutor; only used to resolve some
        relative paths internally. Defaults to "<inline>" since we pass the
        config dict directly.
    """
    disp = build_dispatcher(
        workflow_path=workflow_path,
        comp_dir=components_dir,
        run_tag=run_tag,
        state_dir=component_state_dir,
    )

    ctx = ComponentContext(
        benchmark=domain,
        task_id=task_id,
        user_info=task_config_dict.get("user_info", {}) or {},
        gym_servers=list(task_config_dict.get("gym_servers_config", []) or []),
        selected_tools=list(task_config_dict.get("selected_tools", []) or []),
        verifiers=list(task_config_dict.get("verifiers", []) or []),
        system_prompt=task_config_dict.get("system_prompt", ""),
        user_prompt=task_config_dict.get("user_prompt", ""),
    )
    # Phase C: wire ctx.chat / ctx.emit for the locked SUT model.
    disp.wire_capabilities(ctx)

    # Tier-1 task_received: lifecycle anchor before any mount fires.
    disp.emit("task_received", ctx)
    if ctx.blocked:
        return _blocked_result(domain, task_id, ctx)

    # SESSION_START -----------------------------------------------------
    disp.emit("session_start", ctx)
    if ctx.blocked:
        return _blocked_result(domain, task_id, ctx)

    # PRE_PROMPT_BUILD --------------------------------------------------
    disp.emit("pre_prompt_build", ctx)
    if ctx.blocked:
        return _blocked_result(domain, task_id, ctx)
    # Tier-1 pre_context_build alias for cross-sibling consistency.
    disp.emit("pre_context_build", ctx)
    if ctx.blocked:
        return _blocked_result(domain, task_id, ctx)
    # Tier-1 pre_agent_construct: last chance to influence the inference
    # request shape before the BenchmarkConfig + orchestrator are sealed.
    disp.emit("pre_agent_construct", ctx)
    if ctx.blocked:
        return _blocked_result(domain, task_id, ctx)

    # Build upstream BenchmarkConfig from the (possibly mutated) dict ----
    cfg_data = dict(task_config_dict)
    cfg_data["system_prompt"] = ctx.system_prompt
    cfg_data["user_prompt"] = ctx.user_prompt
    cfg_data.setdefault("number_of_runs", 1)
    cfg_data.setdefault("verifiers", [])
    cfg_data.pop("__task_id__", None)
    cfg_data.pop("__domain__", None)
    _resolve_seed_paths(cfg_data)

    # BenchmarkConfig is a plain dataclass — keep only its declared fields.
    allowed = {f for f in BenchmarkConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    bc_kwargs = {k: v for k, v in cfg_data.items() if k in allowed}
    config = BenchmarkConfig(**bc_kwargs)

    executor = BenchmarkExecutor(
        config,
        llm_config=llm_config,
        orchestrator_class=EnterpriseOpsReactOrchestrator,
        orchestrator_kwargs={"dispatcher": disp, "ctx": ctx},
        config_path=config_path,
    )
    result = await executor.execute_benchmark()

    # PRE_FINAL_EMIT (observational — SQL verifiers already ran upstream) ---
    runs = result.get("runs") or []
    if runs:
        ctx.final_output = (runs[0].get("final_response") or "")[:8000]
    disp.emit("pre_final_emit", ctx)
    # "session_end" dispatches via mount.value = "session_end" which is
    # the same string as the Tier-1 event, so only one emit is needed.
    disp.emit("session_end", ctx)

    # Surface any component side-channel state into the result for downstream
    # inspection (does not affect verifier outcome).
    result["component_trace"] = {
        "blocked": ctx.blocked,
        "blocked_reason": ctx.blocked_reason,
        "n_executed_tool_calls": len(ctx.executed_tool_calls),
        "ctx_shared": {k: v for k, v in ctx.shared.items()
                       if isinstance(v, (str, int, float, bool))},
    }
    return result


def _blocked_result(domain: str, task_id: str, ctx: ComponentContext) -> Dict[str, Any]:
    """Synthetic result for tasks blocked before the executor ran."""
    return {
        "benchmark_config": {"domain": domain, "task_id": task_id},
        "runs": [{
            "run_number": 1,
            "error": f"blocked_pre_executor: {ctx.blocked_reason}",
            "overall_success": False,
        }],
        "statistics": {
            "total_runs": 1,
            "successful_runs": 0,
            "overall_success_rate": 0.0,
            "pass_at_1": 0.0,
            "verifier_level_pass_rate": 0.0,
        },
        "component_trace": {
            "blocked": True,
            "blocked_reason": ctx.blocked_reason,
            "n_executed_tool_calls": 0,
        },
    }


def run_task_sync(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Synchronous convenience wrapper around `run_task`."""
    return asyncio.run(run_task(*args, **kwargs))
