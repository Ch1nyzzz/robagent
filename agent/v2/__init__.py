"""GAIA agent v2 — softer routing: retrieval-flagged tasks still hit LLM.

v1 routed NEEDS_RETRIEVAL tasks to BLOCKED, sacrificing the LLM's parametric
knowledge. v2 keeps NEEDS_FILE blocked (no recovery without a reader) but
routes NEEDS_RETRIEVAL tasks to a separate LLM node that explicitly tells
the model it can answer from prior knowledge or emit `UNKNOWN`.
"""
