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
from pathlib import Path
from types import ModuleType


def load_module_from_path(path: Path, pkg_prefix: str) -> ModuleType:
    """Load a Python module from `path` and register it under
    `<pkg_prefix>.<stem>` in `sys.modules` (idempotent).

    pkg_prefix examples:
      - gaia / tau2 (legacy shared dir):  "agent.components"
      - sopbench (per-domain):            "agent.components_sopbench_<domain>"
      - enterpriseops:                    "agent.components_enterpriseops_<domain>"
      - tau2 v2:                          "agent_tau2.components"
      - toolathlon:                       "agent_toolathlon.components"
    """
    pkg_name = f"{pkg_prefix}.{path.stem}"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(pkg_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod
