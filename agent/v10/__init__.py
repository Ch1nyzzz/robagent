"""v10 — adds gate_blocked_route_calibration and finalizes the harness for full-165 run.

Inherits the v9 workflow (Wikipedia retrieval, file reader, vision, token-aware
rate limit, deterministic query fallback) and adds one more calibration gate:
NEEDS_FILE tasks with file extensions that v8+ supports must NOT BLOCK with
`needs_file_capability` — that would indicate a path resolution bug, not a
capability gap.
"""
