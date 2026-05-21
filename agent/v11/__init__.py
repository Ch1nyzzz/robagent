"""v11 — deterministic refinements on top of v10.

Changes vs v10 (no LLM-capability changes, only deterministic code):
- agent.llm.chat() uses a strict pre-acquisition queue (60 qpm / 1M tpm,
  no fail-fast retry wrapper). Targets v10's 41 RetryError failures.
- route_by_extras_v11 catches implicit-retrieval signals v10 missed:
  Wikipedia / ArXiv / YouTube / restaurant + date / paper-title-in-quotes /
  cipher prompts. Targets v10's 38 wrong-DIRECT failures.
- normalize_answer_v11 fixes format drift on numbers with internal whitespace,
  ordinals, full-width digits. Targets a subset of v10's regressions.
- extract_final_answer_strict trims literal-instruction-only outputs.
"""
