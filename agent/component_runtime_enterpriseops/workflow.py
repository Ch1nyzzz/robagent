"""Workflow + Patch ops + YAML serdes for EnterpriseOps-Gym.

Sibling of `agent/component_runtime_sopbench/workflow.py` (SOP-Bench) and
the GAIA / tau2 variants. Kept as a duplicate (not shared) so each runtime
can extend the patch / edge vocabulary independently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


@dataclass(frozen=True)
class Workflow:
    nodes: tuple[str, ...] = ()
    edges: tuple[Edge, ...] = ()
    disabled: frozenset[str] = field(default_factory=frozenset)

    def active_nodes(self) -> tuple[str, ...]:
        return tuple(n for n in self.nodes if n not in self.disabled)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Workflow":
        text = Path(path).read_text()
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
                section = line.split(":", 1)[0].strip()
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            item = stripped[2:].strip()
            if section == "nodes":
                nodes.append(item)
            elif section == "edges":
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


class PatchOp(str, Enum):
    ADD_NODE     = "add_node"
    REPLACE_NODE = "replace_node"
    DISABLE_NODE = "disable_node"


@dataclass(frozen=True)
class Patch:
    op: PatchOp
    name: str
    file: Optional[str] = None
    edges_in: tuple[Edge, ...] = ()
    edges_out: tuple[Edge, ...] = ()

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


@dataclass
class FrontierSnapshot:
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
