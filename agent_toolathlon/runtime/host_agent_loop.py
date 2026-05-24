"""Robagent host-side decoupled agent loop runner.

Drop-in replacement for `scripts.decoupled.host_agent_loop` (the Toolathlon
default). It re-uses every helper from that module — the only difference
is that the TaskAgent instance is produced by an external candidate
factory selected via the `TOOLATHLON_CANDIDATE` environment variable.

The candidate module path is `agent_toolathlon.<TOOLATHLON_CANDIDATE>.agent`
and must expose `build_agent(**kwargs) -> TaskAgent`. v0 returns the
PrettyDecoupledTaskAgent unchanged, so behaviour matches stock Toolathlon
for the baseline; future candidates can inject component-runtime hooks via
`agent_hooks` / `run_hooks` arguments.

Wired into Toolathlon-src's run_single_decoupled.sh by setting:

    TOOLATHLON_HOST_LOOP_MODULE=agent_toolathlon.runtime.host_agent_loop

(see the small `[robagent patch]` block in that shell script).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import traceback
from functools import partial
from typing import Any

# Re-use ALL helpers from the upstream module — including the pretty
# console printers, MCPServer-config builder, and termination checker.
from scripts.decoupled.host_agent_loop import (  # type: ignore
    ANSI_GREEN,
    ANSI_MAGENTA,
    ANSI_RED,
    PrettyDecoupledTaskAgent,
    build_gateway_runtime_mcp_config,
    build_host_task_config,
    decoupled_termination_checker,
    expand_stop_tool_names,
    print_log_line,
    print_session_header,
    read_json_file,
    preview_text,
)

from utils.general.helper import (  # type: ignore
    build_agent_model_provider,
    build_user_client,
    setup_proxy,
)
from utils.roles.task_agent import TaskStatus  # type: ignore
from utils.task_runner.runner import TaskRunner  # type: ignore

# Required to keep upstream monkey-patches active in our flow too.
from utils.openai_agents_monkey_patch.custom_run_impl import *  # noqa: F401,F403
from utils.openai_agents_monkey_patch.custom_mcp_util import *  # noqa: F401,F403


def _resolve_candidate_factory() -> Any:
    """Resolve `agent_toolathlon.<CANDIDATE>.agent::build_agent`.

    Falls back to v0 if `TOOLATHLON_CANDIDATE` is unset.
    """
    name = os.environ.get("TOOLATHLON_CANDIDATE", "v0").strip() or "v0"
    mod = importlib.import_module(f"agent_toolathlon.{name}.agent")
    if not hasattr(mod, "build_agent"):
        raise RuntimeError(
            f"Candidate '{name}' does not expose build_agent(...)"
        )
    return mod.build_agent, name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robagent host decoupled agent loop")
    p.add_argument("--bundle_file", required=True)
    p.add_argument("--gateway_url", required=True)
    p.add_argument("--gateway_server_name", default="gw")
    p.add_argument("--with_proxy", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--allow_resume", action="store_true")
    return p.parse_args()


async def run_host_loop(args: argparse.Namespace) -> int:
    bundle = read_json_file(args.bundle_file)
    setup_proxy(args.with_proxy)

    eval_config_dict = bundle["eval_config"]
    mcp_config, agent_config, user_config = TaskRunner.load_configs(eval_config_dict)
    task_config = build_host_task_config(bundle, agent_short_name=agent_config.model.short_name)

    runtime_dir = os.path.join(task_config.task_root, ".decoupled_runtime")
    mcp_config.server_config_path = build_gateway_runtime_mcp_config(
        runtime_dir=runtime_dir,
        gateway_server_name=args.gateway_server_name,
        gateway_url=args.gateway_url,
    )
    task_config.needed_mcp_servers = [args.gateway_server_name]

    task_config.stop.tool_names = expand_stop_tool_names(
        stop_tools=task_config.stop.tool_names or ["local-claim_done"],
        gateway_server_name=args.gateway_server_name,
    )

    agent_model_provider = build_agent_model_provider(agent_config)
    user_client = build_user_client(user_config)

    build_agent, candidate_name = _resolve_candidate_factory()

    print_session_header(
        model_name=agent_config.model.short_name,
        gateway_url=args.gateway_url,
        workspace=task_config.agent_workspace,
    )
    print_log_line("CAND", f"candidate={candidate_name}", ANSI_MAGENTA)
    print_log_line("USER", preview_text(task_config.task_str), ANSI_MAGENTA)

    termination_checker = partial(
        decoupled_termination_checker,
        user_stop_phrases=task_config.stop.user_phrases,
        agent_stop_tools=task_config.stop.tool_names,
    )

    task_agent = build_agent(
        task_config=task_config,
        agent_config=agent_config,
        agent_model_provider=agent_model_provider,
        user_config=user_config,
        user_client=user_client,
        mcp_config=mcp_config,
        termination_checker=termination_checker,
        debug=False,
        allow_resume=args.allow_resume,
        single_turn_mode=task_config.single_turn_mode,
        # `pretty_base` lets future candidates extend the printing-enabled
        # base class — v0 already returns PrettyDecoupledTaskAgent.
        pretty_base=PrettyDecoupledTaskAgent,
    )

    current_dir = os.path.abspath(os.getcwd())
    task_status = TaskStatus.FAILED

    try:
        task_agent.status_manager.update_preprocess("done")
        await task_agent.setup_mcp_servers(
            local_token_key_session=bundle.get("local_token_key_session")
        )
        await task_agent.setup_agent()
        await task_agent.setup_user_simulator()

        os.chdir(task_config.agent_workspace)
        task_agent.status_manager.update_running("running")
        await task_agent.run_interaction_loop(
            abs_original_task_root=os.path.abspath(task_config.task_root)
        )

        if task_agent.task_status not in [TaskStatus.MAX_TURNS_REACHED, TaskStatus.INTERRUPTED]:
            task_status = TaskStatus.SUCCESS
            task_agent.status_manager.update_running("done")
        else:
            task_status = task_agent.task_status
            if task_status == TaskStatus.MAX_TURNS_REACHED:
                task_agent.status_manager.update_running("max_turn_exceeded")

    except Exception as e:
        if args.debug:
            traceback.print_exc()
        task_status = TaskStatus.FAILED
        task_agent.status_manager.update_running("fail")
        print(f"Host loop failed: {e}")
    finally:
        os.chdir(current_dir)
        task_agent.task_status = task_status
        user_cost, agent_cost = task_agent.get_cost_summary()
        task_agent.user_cost = user_cost
        task_agent.agent_cost = agent_cost
        await task_agent.save_results()
        await task_agent.cleanup()

    color = ANSI_GREEN if task_status == TaskStatus.SUCCESS else ANSI_RED
    print_log_line(
        "RESULT",
        (
            f"status={task_status.value} "
            f"user_turns={task_agent.stats['interaction_turns']} "
            f"tool_calls={task_agent.stats['tool_calls']} "
            f"agent_requests={task_agent.stats['agent_llm_requests']}"
        ),
        color,
    )
    print(f"Host loop completed with status: {task_status.value}")
    return 0 if task_status == TaskStatus.SUCCESS else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run_host_loop(args)))


if __name__ == "__main__":
    main()
