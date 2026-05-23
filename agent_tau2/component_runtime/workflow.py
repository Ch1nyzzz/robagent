"""Workflow graph + Patch ops + YAML serdes.

A `Workflow` is the frontier: an ordered sequence of Component names plus
edges and a disabled set. The runtime loads the workflow, looks up each
active name in the component registry, and dispatches by mount.

The 3 patch ops:

  * `add_node(name, edges_in, edges_out)`   — append a new node
  * `replace_node(name)`                    — keep id, file overwrites in place
  * `disable_node(name)`                    — set node.enabled=false; file remains

`wrap_tool` / `insert_before` are sugar derivable from `add_node` plus an
explicit `edges_in` / matcher pattern — not exposed as separate ops.

v1 dispatch order is `(priority ASC, insertion ASC)` per mount; edges are
stored in YAML for visualisation / v2 graph traversal but do NOT
participate in v1 dispatch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# --- workflow primitives -----------------------------------------------------


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


@dataclass(frozen=True)
class Workflow:
    """The frontier graph.

    `nodes` is a tuple of Component names in insertion order.
    `edges` is a tuple of Edge records (v1: declarative only).
    `disabled` names a subset of `nodes` to skip at dispatch.
    """
    nodes: tuple[str, ...] = ()
    edges: tuple[Edge, ...] = ()
    disabled: frozenset[str] = field(default_factory=frozenset)

    def active_nodes(self) -> tuple[str, ...]:
        return tuple(n for n in self.nodes if n not in self.disabled)

    # ---- YAML serdes ---------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Workflow":
        """Load a workflow from a YAML file.

        We use a hand-rolled minimal YAML parser (limited to the subset our
        schema needs) to avoid taking a hard dependency on PyYAML. The
        schema is:

            nodes:
              - name_a
              - name_b
            edges:
              - {src: name_a, dst: name_b}
            disabled:
              - name_x
        """
        text = Path(path).read_text() if not isinstance(path, str) else Path(path).read_text()
        return cls._parse_yaml(text)

    @classmethod
    def _parse_yaml(cls, text: str) -> "Workflow":
        nodes: list[str] = []
        edges: list[Edge] = []
        disabled: list[str] = []
        section: Optional[str] = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1].strip()
                continue
            if line.endswith("[]"):
                # empty list, e.g. `edges: []`
                section = line.split(":", 1)[0].strip()
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                # ignore non-list lines (e.g. inline empty objects)
                continue
            item = stripped[2:].strip()
            if section == "nodes":
                nodes.append(item)
            elif section == "edges":
                # Accept `{src: X, dst: Y}` inline-object form.
                src, dst = _parse_edge_item(item)
                edges.append(Edge(src=src, dst=dst))
            elif section == "disabled":
                disabled.append(item)
        return cls(
            nodes=tuple(nodes),
            edges=tuple(edges),
            disabled=frozenset(disabled),
        )

    def to_yaml(self, path: Path | str) -> None:
        """Dump the workflow to a YAML file in canonical form."""
        lines: list[str] = []
        if self.nodes:
            lines.append("nodes:")
            for n in self.nodes:
                lines.append(f"  - {n}")
        else:
            lines.append("nodes: []")
        if self.edges:
            lines.append("edges:")
            for e in self.edges:
                lines.append(f"  - {{src: {e.src}, dst: {e.dst}}}")
        else:
            lines.append("edges: []")
        if self.disabled:
            lines.append("disabled:")
            for n in sorted(self.disabled):
                lines.append(f"  - {n}")
        else:
            lines.append("disabled: []")
        Path(path).write_text("\n".join(lines) + "\n")


def _parse_edge_item(item: str) -> tuple[str, str]:
    """Parse `{src: A, dst: B}` (with optional spaces / quotes)."""
    s = item.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    src = dst = ""
    for kv in s.split(","):
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        k = k.strip().strip("'\"")
        v = v.strip().strip("'\"")
        if k == "src":
            src = v
        elif k == "dst":
            dst = v
    if not src or not dst:
        raise ValueError(f"malformed edge entry: {item!r}")
    return src, dst


# --- patch -------------------------------------------------------------------


class PatchOp(str, Enum):
    ADD_NODE     = "add_node"
    REPLACE_NODE = "replace_node"
    DISABLE_NODE = "disable_node"


@dataclass(frozen=True)
class Patch:
    op: PatchOp
    name: str                                # component name affected
    file: Optional[str] = None               # ADD/REPLACE: rel path to .py
    edges_in: tuple[Edge, ...] = ()          # ADD: edges TO this node
    edges_out: tuple[Edge, ...] = ()         # ADD: edges FROM this node

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        return cls(
            op=PatchOp(d["op"]),
            name=d.get("name") or d.get("node_id") or "",
            file=d.get("file"),
            edges_in=tuple(Edge(**e) for e in d.get("edges_in", []) or []),
            edges_out=tuple(Edge(**e) for e in d.get("edges_out", []) or []),
        )

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "name": self.name,
            "file": self.file,
            "edges_in": [{"src": e.src, "dst": e.dst} for e in self.edges_in],
            "edges_out": [{"src": e.src, "dst": e.dst} for e in self.edges_out],
        }


def apply_patch(wf: Workflow, patch: Patch) -> Workflow:
    """Pure function: return a new Workflow with the patch applied."""
    if patch.op is PatchOp.ADD_NODE:
        if patch.name in wf.nodes:
            raise ValueError(f"add_node: {patch.name!r} already exists in workflow")
        return Workflow(
            nodes=wf.nodes + (patch.name,),
            edges=wf.edges + patch.edges_in + patch.edges_out,
            disabled=wf.disabled,
        )
    if patch.op is PatchOp.REPLACE_NODE:
        if patch.name not in wf.nodes:
            raise ValueError(f"replace_node: {patch.name!r} not in workflow")
        # Node ordering and edges preserved; the new file under the same
        # name takes the same slot at dispatch (registry dedups by name).
        return wf
    if patch.op is PatchOp.DISABLE_NODE:
        if patch.name not in wf.nodes:
            raise ValueError(f"disable_node: {patch.name!r} not in workflow")
        return Workflow(
            nodes=wf.nodes,
            edges=wf.edges,
            disabled=wf.disabled | {patch.name},
        )
    raise ValueError(f"unknown patch op: {patch.op!r}")


# --- frontier snapshot -------------------------------------------------------


@dataclass
class FrontierSnapshot:
    """The JSON-serializable snapshot the scoring loop trusts.

    Written only on accept, in `meta_harness/logs_tau2_components/frontier_workflow.json`.
    Distinct from the human-edited YAML on disk: this captures the byte-text
    of the YAML at the moment of acceptance so a rejected later attempt
    cannot silently shift the "previous frontier" the scoring compares to.
    """
    workflow_yaml_at_accept: str
    active_names: list[str]
    accepted_at_iteration: int
    accepted_at: str

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({
            "workflow_yaml_at_accept": self.workflow_yaml_at_accept,
            "active_names": self.active_names,
            "accepted_at_iteration": self.accepted_at_iteration,
            "accepted_at": self.accepted_at,
        }, indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "FrontierSnapshot":
        d = json.loads(Path(path).read_text())
        return cls(
            workflow_yaml_at_accept=d["workflow_yaml_at_accept"],
            active_names=list(d["active_names"]),
            accepted_at_iteration=int(d["accepted_at_iteration"]),
            accepted_at=str(d["accepted_at"]),
        )
