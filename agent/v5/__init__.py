"""GAIA agent v5 — consolidation.

v5 is identical to v4 in behavior but with a stricter answer extractor for
the case where the LLM emits a verbose, scaffolded response despite the prompt.
Adds a `gate_route_consistency` audit and elevates the workflow definition
to a single canonical place.
"""
