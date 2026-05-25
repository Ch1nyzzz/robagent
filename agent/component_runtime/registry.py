"""Load GAIA Component modules.

Each Component lives in `agent/components/<name>.py` and exports
`COMPONENT: Component`. Mirrors the tau2 registry; later files with the
same `COMPONENT.name` replace earlier ones (the `replace_node` semantics
at the file level).

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


COMPONENTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "components"
_PKG_PREFIX = "agent.components"


def load_components(component_files: Iterable[str | Path]) -> dict[Mount, list[Component]]:
    by_name: dict[str, Component] = {}
    for raw in component_files:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"component file not found: {path}")
        mod = load_module_from_path(path, _PKG_PREFIX)
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
