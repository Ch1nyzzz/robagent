"""Load hook modules into a registry.

Each hook lives in its own file under `agent_tau2/hooks/<name>.py` and
exports a module-level constant `HOOK: Hook`. The registry is built by
loading a set of file paths (the "frontier hook set") and validating each
registration against the class×event permission matrix.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

from .policy import validate_registration
from .types import Hook, HookEvent


HOOKS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "hooks"


def _load_module_from_path(path: Path):
    """Import a hook file as `agent_tau2.hooks.<stem>` so relative imports
    inside hook modules still work, but without requiring the file to live
    on `sys.path`."""
    pkg_name = f"agent_tau2.hooks.{path.stem}"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(pkg_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load hook module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_hooks(hook_files: Iterable[str | Path]) -> dict[HookEvent, list[Hook]]:
    """Load a set of hook files. Later files with the same HOOK.name replace
    earlier ones (the modify-by-reuse-name convention)."""
    by_name: dict[str, Hook] = {}
    for raw in hook_files:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"hook file not found: {path}")
        mod = _load_module_from_path(path)
        if not hasattr(mod, "HOOK"):
            raise AttributeError(f"hook module {path} must export `HOOK`")
        hook: Hook = mod.HOOK
        validate_registration(hook.cls, hook.event)
        by_name[hook.name] = hook

    grouped: dict[HookEvent, list[Hook]] = {e: [] for e in HookEvent}
    for hook in by_name.values():
        grouped[hook.event].append(hook)
    return grouped


def load_hooks_from_dir(directory: str | Path = HOOKS_DIR_DEFAULT,
                        only: Iterable[str] | None = None
                        ) -> dict[HookEvent, list[Hook]]:
    """Load every `*.py` under directory (excluding dunder files). If `only`
    is given, restrict to hook NAMES in that set (the per-iteration filter
    applied by meta_harness_hooks.py)."""
    directory = Path(directory)
    files = sorted(p for p in directory.glob("*.py")
                   if not p.name.startswith("_"))
    grouped = load_hooks(files)
    if only is None:
        return grouped
    only_set = set(only)
    return {
        event: [h for h in hooks if h.name in only_set]
        for event, hooks in grouped.items()
    }
