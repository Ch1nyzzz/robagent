"""v14 tools — overlay vision_describe with the rate-slot-fixed version.

Everything else is re-exported unchanged from v13.
"""
from agent.v13.tools import (  # noqa: F401
    arxiv_search,
    audio_read,
    duckduckgo_html_search,
    gaia_file_path,
    pdb_read,
    read_extra_file,
    read_gaia_file,
    wikipedia_fetch,
    wikipedia_search,
    zip_read,
)

from .vision_v14 import vision_describe  # noqa: F401

__all__ = [
    "arxiv_search",
    "audio_read",
    "duckduckgo_html_search",
    "gaia_file_path",
    "pdb_read",
    "read_extra_file",
    "read_gaia_file",
    "vision_describe",
    "wikipedia_fetch",
    "wikipedia_search",
    "zip_read",
]
