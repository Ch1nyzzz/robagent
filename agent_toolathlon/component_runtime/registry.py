"""Load Component modules into a registry.

Each Component lives in its own file under
`agent_toolathlon/components/<name>.py` and exports a module-level
constant `COMPONENT: Component`. The registry is built by loading a set
of file paths (the active workflow's node set) and validating each
registration against the class×mount permission matrix plus the trust
profile.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

from .policy import validate_registration, validate_trust
from .types import Component, Mount


COMPONENTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "components"


def _load_module_from_path(path: Path):
    """Import a component file as `agent_toolathlon.components.<stem>` so
    relative imports inside component modules still work, but without
    requiring the file to live on `sys.path`."""
    pkg_name = f"agent_toolathlon.components.{path.stem}"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(pkg_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_components(component_files: Iterable[str | Path]) -> dict[Mount, list[Component]]:
    """Load a set of component files. Later files with the same
    COMPONENT.name replace earlier ones (modify-by-reuse-name — this is
    how `replace_node` is realised at the filesystem layer)."""
    by_name: dict[str, Component] = {}
    for raw in component_files:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"component file not found: {path}")
        mod = _load_module_from_path(path)
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
    """Load every `*.py` under directory (excluding dunder files). If
    `only` is given, restrict to component NAMES in that set (the
    per-iteration filter applied by the outer loop)."""
    directory = Path(directory)
    files = sorted(p for p in directory.glob("*.py")
                   if not p.name.startswith("_"))
    grouped = load_components(files)
    if only is None:
        return grouped
    only_set = set(only)
    return {
        mount: [c for c in comps if c.name in only_set]
        for mount, comps in grouped.items()
    }
