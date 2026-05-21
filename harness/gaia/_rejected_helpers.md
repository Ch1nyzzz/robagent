# Rejected helpers — GAIA iter 1

| Helper | Reason for rejection |
|---|---|
| `parse_answer_type(question) -> {"number","string","list"}` | High false-positive risk. Many GAIA questions don't announce expected format syntactically (e.g. "What did X say at minute Y?" could be a quote or a topic). Without a measurable >=80% accuracy across the corpus we keep this as an LLM-implicit judgement. |
| `extract_number_from_prose(text) -> float` | Conflicts with named entities containing digits (years, room numbers, etc.). Would corrupt non-numeric answers. |
| `coerce_list(answer, separator=",") -> list[str]` | The scorer already handles comma/semicolon splits; an extra coercion would only hide answer-formation bugs. |

Each was AST-validated as containing no rare-corpus tokens, but failed the dynamic empirical check on the firing set — agreement with v0 LLM output was below 80%, so the LLM stays.
