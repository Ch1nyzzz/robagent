"""Load EnterpriseOps-Gym Component modules.

Each Component lives in `agent/components_enterpriseops_<domain>/<name>.py`
and exports `COMPONENT: Component`. Later files with the same
`COMPONENT.name` replace earlier ones (file-level replace_node), mirroring
the sopbench / GAIA / tau2 sibling registries.

The pkg_prefix is dynamic per-domain — `agent.components_enterpriseops.<domain_dir_name>.<stem>`
keeps each domain's components in its own sys.modules namespace so
calendar and itsm runs don't trample each other when both are loaded
in the same process.

Refactored in Phase A of the event-runtime migration to delegate the
file-loading bit to `meta_harness.component_runtime_core.registry`.
Per-sibling behaviour (Mount groupby, ALLOWED policy validation) stays
here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from meta_harness.component_runtime_core.registry import load_module_from_path

from .policy import validate_registration, validate_trust
from .types import Component, Mount


# Default directory is `agent/components_enterpriseops_calendar` purely so
# that bare imports don't error; in practice callers always pass
# `comp_dir` explicitly (one dir per domain).
COMPONENTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "components_enterpriseops_calendar"


def _pkg_prefix_for(path: Path) -> str:
    """Per-domain pkg prefix so sys.modules entries from different
    domains don't collide."""
    return f"agent.components_enterpriseops.{path.parent.name}"


def load_components(component_files: Iterable[str | Path]) -> dict[Mount, list[Component]]:
    by_name: dict[str, Component] = {}
    for raw in component_files:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"component file not found: {path}")
        mod = load_module_from_path(path, _pkg_prefix_for(path))
        if not hasattr(mod, "COMPONENT"):
            raise AttributeError(f"component module {path} must export `COMPONENT`")
        comp: Component = mod.COMPONENT
        validate_registration(comp.cls, comp.mount)
        validate_trust(comp.cls, comp.trust)
        by_name[comp.name] = comp

    grouped: dict[Mount, list[Component]] = {m: [] for m in Mount}
    for comp in by_name.values():
        grouped[comp.mount].append(comp)
    for mount, comps in grouped.items():
        comps.sort(key=lambda c: c.priority)
    return grouped


def load_components_from_dir(directory: str | Path = COMPONENTS_DIR_DEFAULT,
                             only: Iterable[str] | None = None
                             ) -> dict[Mount, list[Component]]:
    directory = Path(directory)
    if not directory.exists():
        return {m: [] for m in Mount}
    files = sorted(p for p in directory.glob("*.py")
                   if not p.name.startswith("_"))
    if not files:
        return {m: [] for m in Mount}
    grouped = load_components(files)
    if only is None:
        return grouped
    only_set = set(only)
    return {
        mount: [c for c in comps if c.name in only_set]
        for mount, comps in grouped.items()
    }
