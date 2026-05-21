"""GAIA agent v1 — restructure baseline into deterministic + LLM nodes with audit gates.

No new capabilities. Routes file/retrieval tasks to BLOCKED with structured reasons,
answers DIRECT tasks via LLM with a max_tokens budget large enough for V4-Pro reasoning.
"""
