"""GAIA agent v3 — v2 + answer-shape normalization tuned to scorer branches.

Builds on v2's parametric-retrieval LLM node. Adds a deterministic post-LLM
reshape step that interprets the question for hints about expected answer
shape (numeric, list, single token) and reformats accordingly, so the
scorer's branch selection matches the agent's intent.
"""
