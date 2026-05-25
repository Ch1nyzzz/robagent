"""Load Component modules into a registry.

Each Component lives in its own file under `agent_tau2/components/<name>.py`
and exports a module-level constant `COMPONENT: Component`. The registry is
built by loading a set of file paths (the active workflow's node set) and
validating each registration against the class×mount permission matrix
plus the trust profile.

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
from .types import Component
COMPONENTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "components"
_PKG_PREFIX = "agent_tau2.components"


def load_components(component_files: Iterable[str | Path]) -> list[Component]:
    """Load a set of component files. Later files with the same COMPONENT.name
    replace earlier ones (the modify-by-reuse-name convention — this is how
    `replace_node` is realised at the filesystem layer)."""
    by_name: dict[str, Component] = {}
    for raw in component_files:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"component file not found: {path}")
        mod = load_module_from_path(path, _PKG_PREFIX)
        if not hasattr(mod, "COMPONENT"):
            raise AttributeError(f"component module {path} must export `COMPONENT`")
        comp: Component = mod.COMPONENT
        validate_registration(comp.cls, comp.listens)
        validate_trust(comp.cls, comp.trust)
        by_name[comp.name] = comp

    # Stable ordering: (priority ASC, insertion ASC). Insertion order
    # is the dict iteration order of `by_name`, which Python 3.7+
    # guarantees. Sorting on priority alone preserves that for ties.
    return sorted(by_name.values(), key=lambda c: c.priority)


def load_components_from_dir(directory: str | Path = COMPONENTS_DIR_DEFAULT,
                             only: Iterable[str] | None = None
                             ) -> list[Component]:
    """Load every `*.py` under directory (excluding dunder files). If `only`
    is given, restrict to component NAMES in that set (the per-iteration
    filter applied by meta_harness_components.py)."""
    directory = Path(directory)
    files = sorted(p for p in directory.glob("*.py")
                   if not p.name.startswith("_"))
    all_comps = load_components(files)
    if only is None:
        return all_comps
    only_set = set(only)
    return [c for c in all_comps if c.name in only_set]
