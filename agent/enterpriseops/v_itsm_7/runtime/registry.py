"""Load EnterpriseOps-Gym Component modules.

Each Component lives in `agent/enterpriseops/<v_dir>/components_<domain>/<name>.py`
and exports `COMPONENT: Component`. Later files with the same `COMPONENT.name`
replace earlier ones.

The pkg_prefix is derived from the file's path so that:
  - relative imports inside a component (e.g. `from ..runtime import Decision`)
    resolve to the runtime in the same v_N tree
  - sibling v_N copies don't share sys.modules entries
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ballast.component_runtime_core.registry import load_module_from_path

from .policy import validate_registration, validate_trust
from .types import Component


def _pkg_prefix_for(path: Path) -> str:
    """Derive the dotted package prefix from the file path.

    Walks up from the file to find the first directory named `agent`, then
    rejoins everything below it. Examples:

      /…/agent/enterpriseops/v0/components_calendar/foo.py
        → "agent.enterpriseops.v0.components_calendar"
      /…/agent/enterpriseops/v_itsm_3/components_itsm/bar.py
        → "agent.enterpriseops.v_itsm_3.components_itsm"

    Components written by the proposer use relative imports
    (`from ..runtime import …`), which resolve correctly because the package
    chain `agent.enterpriseops.<v_N>` exists as real directories with
    __init__.py files.
    """
    parts = path.parent.parts
    for i, p in enumerate(parts):
        if p == "agent":
            return ".".join(parts[i:])
    return ".".join(parts[-4:])


def load_components(component_files: Iterable[str | Path]) -> list[Component]:
    by_name: dict[str, Component] = {}
    for raw in component_files:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"component file not found: {path}")
        mod = load_module_from_path(path, _pkg_prefix_for(path))
        if not hasattr(mod, "COMPONENT"):
            raise AttributeError(f"component module {path} must export `COMPONENT`")
        comp: Component = mod.COMPONENT
        validate_registration(comp.cls, comp.listens)
        validate_trust(comp.cls, comp.trust)
        by_name[comp.name] = comp

    return sorted(by_name.values(), key=lambda c: c.priority)


def load_components_from_dir(directory: str | Path) -> list[Component]:
    """Load every `*.py` (sorted) under `directory`, except `_*.py`.

    Returns [] for a missing or empty directory.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    files = sorted(p for p in directory.glob("*.py")
                   if not p.name.startswith("_"))
    if not files:
        return []
    return load_components(files)
