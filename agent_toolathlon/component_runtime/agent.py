"""Component runtime entry point (build_agent + Workflow resolution).

This is the candidate `agent.py` consumed via TOOLATHLON_CANDIDATE=cr.
The function `build_agent(**kwargs)` is what host_agent_loop.py calls
once per task; it:

  1. Resolves the active Workflow (YAML file → frozen Workflow).
  2. Loads the active Component set from disk (one .py per component).
  3. Builds ComponentAgentHooks + ComponentRunHooks bound to that set.
  4. Returns a `CrPrettyDecoupledTaskAgent` instance with the hooks
     attached and the component-by-mount table wired into CR_HOOK
     dispatch.

Environment-variable contract (mirrors agent_tau2/component_runtime/agent.py):

  COMPONENT_WORKFLOW  path to YAML workflow file
                      (default: meta_harness/workflows/toolathlon_main.yaml)
  COMPONENT_NAMES     comma-separated names; defensive check that the
                      outer loop has the workflow pinned correctly.
                      Empty = trust the workflow YAML.
  COMPONENT_DIR       directory of component .py files
                      (default: agent_toolathlon/components/)
  COMPONENT_FILES     colon-separated explicit paths (overrides
                      dir+workflow; primarily for unit tests).
  COMPONENT_RUN_TAG   namespacing tag for
                      .component-state-toolathlon/<tag>/fired.jsonl
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .hooks import build_hooks
from .pretty_task_agent import PrettyDecoupledTaskAgent
from .registry import COMPONENTS_DIR_DEFAULT, load_components, load_components_from_dir
from .task_agent import TaskAgent
from .types import Component, Mount
from .workflow import Workflow


def _wrap_tools_enabled() -> bool:
    """v2 MCP tool wrapping is enabled by default for `cr` candidate.
    Set COMPONENT_WRAP_TOOLS=0 to fall back to the v1 path (mcp_servers
    given directly to Agent; PRE_TOOL_USE REWRITE / true BLOCK and
    single-turn POST_TOOL_USE inject become unavailable)."""
    v = os.environ.get("COMPONENT_WRAP_TOOLS", "1").strip().lower()
    return v not in ("", "0", "false", "no", "off")


ROOT = Path(__file__).resolve().parents[2]   # robagent/


def _resolve_workflow_and_components() -> tuple[Workflow, dict[str, Component]]:
    files_env = os.environ.get("COMPONENT_FILES", "").strip()
    if files_env:
        grouped = load_components(files_env.split(":"))
        components_by_name = {
            c.name: c for comps in grouped.values() for c in comps
        }
        wf = Workflow(nodes=tuple(components_by_name.keys()))
        return wf, components_by_name

    workflow_path = Path(os.environ.get(
        "COMPONENT_WORKFLOW",
        str(ROOT / "meta_harness" / "workflows" / "toolathlon_main.yaml"),
    ))
    if not workflow_path.exists():
        wf = Workflow()
    else:
        wf = Workflow.from_yaml(workflow_path)

    names_env = os.environ.get("COMPONENT_NAMES", "").strip()
    names_from_env = {n for n in names_env.split(",") if n}
    if names_from_env and names_from_env != set(wf.active_nodes()):
        raise SystemExit(
            f"COMPONENT_NAMES {sorted(names_from_env)} ≠ workflow active "
            f"{sorted(wf.active_nodes())}; the outer loop must apply the "
            f"patch before invoking the runner."
        )

    comp_dir = os.environ.get("COMPONENT_DIR", str(COMPONENTS_DIR_DEFAULT))
    active = list(wf.active_nodes())
    if active:
        grouped = load_components_from_dir(comp_dir, only=active)
        components_by_name = {
            c.name: c for comps in grouped.values() for c in comps
        }
    else:
        components_by_name = {}
    return wf, components_by_name


def _group_by_mount(
    wf: Workflow,
    components_by_name: dict[str, Component],
) -> dict[Mount, list[Component]]:
    """Bucket active components by mount, sorted by (priority, insertion).

    Mirrors the bucketing in agent_tau2/component_runtime/agent.py.
    """
    by_mount: dict[Mount, list[Component]] = {m: [] for m in Mount}
    for name in wf.active_nodes():
        comp = components_by_name.get(name)
        if comp is None:
            continue
        by_mount[comp.mount].append(comp)
    insertion = {n: i for i, n in enumerate(wf.nodes)}
    for mount in Mount:
        by_mount[mount].sort(key=lambda c: (c.priority, insertion.get(c.name, 1_000_000)))
    return by_mount


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
    debug: bool = False,
    manual: bool = False,
    pretty_base: Any = None,   # ignored: we always return our pretty subclass
    **_unused: Any,
) -> TaskAgent:
    wf, components_by_name = _resolve_workflow_and_components()
    by_mount = _group_by_mount(wf, components_by_name)
    session_state: dict[str, dict] = {n: {} for n in wf.active_nodes()}

    # tool_names is filled lazily — at build_agent time the MCP gateway
    # hasn't been connected so we don't know tool names yet. Hooks read
    # `_cr_tool_names_snapshot()` from the live TaskAgent at fire time.
    agent_hooks, run_hooks, dispatcher = build_hooks(
        by_mount=by_mount,
        session_state=session_state,
        domain_policy=getattr(task_config, "task_str", "") or "",
        tool_names=(),
    )

    return PrettyDecoupledTaskAgent(
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
        cr_components_by_mount=by_mount,
        cr_session_state=session_state,
        cr_dispatcher=dispatcher,
        cr_wrap_tools=_wrap_tools_enabled(),
    )
