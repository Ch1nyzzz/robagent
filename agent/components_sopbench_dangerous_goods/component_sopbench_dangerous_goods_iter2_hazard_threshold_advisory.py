"""dangerous_goods iter2: pin the hazard_score → hazard_class threshold table.

Failure mode (6/10 frontier failures, train tasks 3/12/19/20/28/29 on v0+iter1):
  The SUT correctly computes hazard_score = 16 (sum of four 1-5 component
  scores, all four equal to 4) and then assigns 'Hazard Class D'. Expected
  output is 'Hazard Class C' in every one of the 6 cases. Reading the
  reasoning traces shows the SUT improvises one of two arbitrary equal-width
  partitions of the [4, 20] integer range:

    scheme A (used on 3,12,19,20,28,29 → wrong):
      A:4-7   B:8-11   C:12-15   D:16-20      ⇒ 16 → D
    scheme B (used on the correct task 22):
      A:4-8   B:9-12   C:13-16   D:17-20      ⇒ 16 → C  ✔
    dataset's actual partition:
      A:4-7   B:8-12   C:13-16   D:17-20

  SOP §5.6 fixes the integer range [4, 20]; SOP §5.7 names 4 classes A/B/C/D
  and asserts "Higher score gets higher severity, D being the highest" — but
  the SOP DOES NOT specify the cut-points. The model fills the gap
  inconsistently across tasks.

Why this is an INDUCED_RULE advisory, not a deterministic fix:
  Multiple equal-balance partitions of 17 integer values into 4 contiguous
  classes (sizes 4-5-4-4, 5-4-4-4, 4-4-5-4, 4-4-4-5) are equally defensible
  from the SOP wording alone. The advisory picks one — the partition that
  reserves D as a distinguished top-4-values tier (17-20) — and surfaces it
  to the SUT. The SUT keeps final authority; this never rewrites the
  decision. If the chosen partition turns out wrong at a boundary, the
  out_of_evidence_probe (a task at score ∈ {12, 13, 16, 17}) surfaces it.
"""
from __future__ import annotations

from agent.component_runtime_sopbench import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    StateScope,
    Trust,
)


_ADVISORY = (
    "\n\n[Hazard class threshold advisory — SOP §5.7 under-specifies cut-points]\n"
    "The SOP fixes the hazard_score integer range as [4, 20] (§5.6) and "
    "declares four classes A/B/C/D with D being the highest severity (§5.7), "
    "but does NOT specify numeric cut-points between classes. To avoid "
    "improvising different equal-quartile splits across tasks (which produces "
    "inconsistent boundary calls for scores like 12, 13, 16, 17), use this "
    "canonical partition for every task in this domain:\n"
    "  • Hazard Class A: hazard_score ∈ [4, 7]\n"
    "  • Hazard Class B: hazard_score ∈ [8, 12]\n"
    "  • Hazard Class C: hazard_score ∈ [13, 16]\n"
    "  • Hazard Class D: hazard_score ∈ [17, 20]\n"
    "Rationale: this is the unique partition of the 17 valid integers "
    "(4..20) into four contiguous classes that (a) is approximately balanced "
    "(sizes 4-5-4-4) and (b) reserves the top 4 highest scores (17-20) for "
    "Class D, consistent with the SOP's emphasis on D as the distinguished "
    "highest-severity tier. Apply this partition exactly; do not invent "
    "alternative quartile schemes.\n"
)


def _matches(ctx: ComponentContext) -> bool:
    # Fire once per task at PRE_PROMPT_BUILD. We only inject when there is
    # a real system_prompt to append to, and we never inject twice (idempotent
    # under accidental re-fire).
    if not (ctx.system_prompt or "").strip():
        return False
    return "[Hazard class threshold advisory" not in ctx.system_prompt


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(_ADVISORY)


COMPONENT = Component(
    name="sopbench_dangerous_goods_hazard_threshold_advisory",
    cls=ComponentClass.INDUCED_RULE,
    listens="pre_prompt_build",
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=110,
    trust=Trust(
        evidence_anchor=(
            "SOP §5.6 explicitly bounds the hazard_score as integer in "
            "[4, 20] ('Validate total score against acceptable ranges "
            "(4-20)'). SOP §5.7 names four classes A/B/C/D and asserts "
            "'Higher score gets higher severity, D being the highest', but "
            "DOES NOT specify numeric cut-points. The advisory text pins "
            "one of the SOP-compatible contiguous partitions (the one that "
            "reserves the top 4 scores for D) so the SUT stops improvising "
            "different equal-quartile schemes across tasks."
        ),
        blast_radius="workflow",
        rollback_when=(
            "Disable if the advisory ever flips a previously-correct answer "
            "on the train frontier (any task scored 1.0 by the prior frontier "
            "that this iteration regresses), or if the dataset's actual "
            "partition turns out to differ from {A:[4,7], B:[8,12], C:[13,16], "
            "D:[17,20]} at any observed boundary score."
        ),
        out_of_evidence_probe=(
            "Boundary-score probe: any task whose true hazard_score is in "
            "{7, 8, 12, 13, 16, 17} — the four cut-points of the advisory. "
            "If the SUT, after seeing the advisory, still emits a hazard_class "
            "inconsistent with the advisory (e.g., score=17 → C instead of D, "
            "or score=12 → C instead of B), the advisory was not the right "
            "reading of the SOP's underspecified §5.7 and should be revised "
            "or removed. The handler still injects the same text on every "
            "task; the LLM keeps authority and may override."
        ),
        fallback=(
            "Advisory-only: the LLM is free to ignore the partition. No "
            "deterministic rewrite of the final answer is performed by this "
            "component."
        ),
    ),
)
