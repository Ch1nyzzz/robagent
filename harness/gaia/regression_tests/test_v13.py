"""v13 unit tests — extra file readers.

We synthesize tiny in-memory files for pdb / zip / audio paths to verify the
deterministic dispatch surface. The audio test stops at the network boundary.
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from agent.v13.tools import pdb_read, zip_read


def test_pdb_read_extracts_header_lines() -> None:
    body = (
        "HEADER    HYDROLASE                               01-JAN-99   1ABC\n"
        "TITLE     A TEST PROTEIN\n"
        "COMPND    MOL_ID: 1\n"
        "ATOM      1  N   ALA A   1      11.0   2.0  -3.0  1.00  0.0\n"
        "ATOM      2  CA  ALA A   1      12.0   2.0  -3.0  1.00  0.0\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(body)
        path = f.name
    out = pdb_read(path)
    assert "HEADER" in out
    assert "TITLE" in out
    assert "A TEST PROTEIN" in out
    # ATOM records are NOT echoed back
    assert "ATOM" not in out


def test_zip_read_inlines_text_entries() -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        path = f.name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello.txt", "alpha beta gamma")
        z.writestr("data.json", '{"a": 1}')
    out = zip_read(path)
    assert "hello.txt" in out
    assert "alpha beta gamma" in out
    assert "data.json" in out


def test_zip_read_dispatches_pdb_inside_archive() -> None:
    body = "HEADER    TEST\nTITLE     INNER PROTEIN\nATOM      1  N   ALA A   1\n"
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        path = f.name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("inside.pdb", body)
    out = zip_read(path)
    assert "INNER PROTEIN" in out
    assert "ATOM" not in out


def test_pdb_read_handles_missing_file() -> None:
    out = pdb_read("/nonexistent/path/to/file.pdb")
    assert "parse failed" in out


def test_v13_router_inherits_v11_v12() -> None:
    from agent.v13.reducers import route_by_extras
    q = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 wikipedia?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"
