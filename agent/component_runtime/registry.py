"""Load GAIA Component modules.

Each Component lives in `agent/components/<name>.py` and exports
`COMPONENT: Component`. Later files with the same `COMPONENT.name`
replace earlier ones (the `replace_node` semantics at the file level).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from meta_harness.component_runtime_core.registry import load_module_from_path

from .policy import validate_registration, validate_trust
from .types import Component


COMPONENTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "components"
_PKG_PREFIX = "agent.components"


def load_components(component_files: Iterable[str | Path]) -> list[Component]:
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
    # Stable sort by (priority, insertion order — dict preserves insertion).
    return sorted(by_name.values(), key=lambda c: c.priority)


def load_components_from_dir(directory: str | Path = COMPONENTS_DIR_DEFAULT,
                             only: Iterable[str] | None = None
                             ) -> list[Component]:
    directory = Path(directory)
    if not directory.exists():
        return []
    files = sorted(p for p in directory.glob("*.py")
                   if not p.name.startswith("_"))
    if not files:
        return []
    all_comps = load_components(files)
    if only is None:
        return all_comps
    only_set = set(only)
    return [c for c in all_comps if c.name in only_set]
