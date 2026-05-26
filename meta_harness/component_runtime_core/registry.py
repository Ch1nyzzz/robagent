"""Generic component module loader.

The 5 sibling registries each contained a near-identical
`_load_module_from_path` plus a `load_components_from_dir` wrapper. The
file-loading bit is generic; the only sibling-specific input is the
package prefix used when registering the module under
`sys.modules[<pkg_prefix>.<stem>]`. We lift the file-loading bit here and
let each sibling supply its own prefix + Mount enum for grouping.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType


# Global serialisation for component module loads. The race we're guarding
# against: thread A registers the freshly-created (empty) module in
# sys.modules before exec_module populates it; thread B sees the key in
# sys.modules, returns the empty module, and `hasattr(mod, "COMPONENT")`
# is False. Symptom under `run_benchmark.py --parallel N>1`: random
# `AttributeError: must export COMPONENT` on the first task that races.
# Loads are infrequent (per task) and cheap once cached, so a single
# global lock is fine.
_LOAD_LOCK = threading.Lock()


def load_module_from_path(path: Path, pkg_prefix: str) -> ModuleType:
    """Load a Python module from `path` and register it under
    `<pkg_prefix>.<stem>` in `sys.modules` (idempotent, thread-safe).

    pkg_prefix examples:
      - gaia / tau2 (legacy shared dir):  "agent.components"
      - sopbench (per-domain):            "agent.components_sopbench_<domain>"
      - enterpriseops:                    "agent.components_enterpriseops_<domain>"
      - tau2 v2:                          "agent_tau2.components"
      - toolathlon:                       "agent_toolathlon.components"
    """
    pkg_name = f"{pkg_prefix}.{path.stem}"
    cached = sys.modules.get(pkg_name)
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        cached = sys.modules.get(pkg_name)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(pkg_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load component module from {path}")
        mod = importlib.util.module_from_spec(spec)
        # Execute BEFORE publishing so other threads never see a half-initialised
        # module. If exec_module raises, the partial module never enters sys.modules.
        spec.loader.exec_module(mod)
        sys.modules[pkg_name] = mod
        return mod
