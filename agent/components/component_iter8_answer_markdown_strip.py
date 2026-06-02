"""Mechanism layer: strip leaked Markdown emphasis wrappers from ctx.answer.

OBSERVED FAILURE
----------------
On iter7 task 50ec8903 (Rubik's-cube colour pair), the LLM ended its
response with the literal line `**FINAL ANSWER: green, white**` — the
entire FINAL ANSWER line wrapped in CommonMark bold. The published
extractor `agent.base._extract_final_answer` uses the regex
`r"FINAL ANSWER:\s*(.+?)\s*$"` with re.IGNORECASE | re.MULTILINE. The
literal `FINAL ANSWER:` anchor starts AFTER the opening `**`, so the
opening wrapper sits outside the capture group; the closing `**` is
inside, because the lazy `.+?` is grown to the last non-whitespace char
before `\\s*$`. Result: extracted answer = `green, white**`. The GAIA
scorer compares against `green, white` and awards 0. Identical pattern
seen on iter2 task 4d0aa727 (`1 in 3**`). Both tasks already score 1.0
under other runs in the frontier; the leak is a variance source, not a
hard task-difficulty failure.

ANCHOR (outside evidence)
-------------------------
(1) CommonMark / GFM Markdown spec §6.4 defines `**…**`, `__…__`,
    `~~…~~`, and `` `…` `` as symmetric inline emphasis/code delimiters
    around span content — a stable external contract.
(2) agent/base.py:44 `_FINAL_RE` — the published extraction contract.
    Reading the regex semantics directly tells you the leak shape: the
    opening multi-char wrapper is excluded (before the literal anchor),
    the closing one is captured (lazy `.+?` extended to `\\s*$`).
(3) agent/base.py:30-35 SYSTEM_PROMPT explicitly mandates plain-text
    answers ("strings: no extra prefix or explanation"); the GAIA
    scorer compares against unwrapped ground truth, so any Markdown
    emphasis in ctx.answer is always wrong by construction.

The mechanism (normalise ctx.answer by stripping orphan / symmetric
Markdown emphasis wrappers at pre_answer_emit) is a general formatting
normalisation independent of any task content.

POLICY MATRIX
-------------
mechanism_layer @ pre_answer_emit admits {allow, rewrite, block}. This
hook returns only allow or rewrite.

SAFETY ENVELOPE
---------------
- Symmetric wrappers (`**…**`, `__…__`, `~~…~~`, `` `…` ``) are
  stripped iteratively only when the answer starts AND ends with the
  same multi-char (or single backtick) wrapper and has length > 2*|w|.
- Asymmetric trailing / leading wrappers are stripped only for the
  multi-char wrappers (`**`, `__`, `~~`) — and only when the wrapper
  does NOT appear inside the remaining span. This protects legitimate
  internal markup (e.g. an answer that happens to contain `**` as part
  of its content).
- Single-char wrappers (`*`, `_`) are never stripped asymmetrically:
  they are too commonly legitimate (math operators, footnote markers,
  identifier suffixes).
- Empty / whitespace answers fall through unchanged.

INTERACTION WITH exhaustion_answer_recovery (priority=110)
----------------------------------------------------------
This hook is priority=130 so it fires AFTER exhaustion_answer_recovery
on the same pre_answer_emit event. If the recovery LLM itself produces
a wrapped FINAL ANSWER line, the recovery's own `_extract_final_answer`
call leaks the closing wrapper into ctx.answer; this hook then cleans
it up. The matchers are non-overlapping anyway (recovery fires on empty
ctx.answer; this hook fires on wrapper-bearing ctx.answer), so the
ordering is conservative.

OUT-OF-EVIDENCE PROBE
---------------------
A: answer="42"            → no wrapper chars; matcher False; no change.
B: answer="**42**"        → symmetric strip → "42".
C: answer="5 * 7 = 35"    → no edge multi-char wrapper; matcher False.
D: answer="green, white**"→ trailing strip → "green, white" (observed).
E: answer=""              → matcher False; no change.
F: answer="`code`"        → symmetric strip → "code".
G: answer="foo*"          → single-char `*` not in multi-char set;
                            matcher False; no change (preserves
                            legitimate trailing asterisks).
H: answer="a **b** c"     → trailing/leading wrapper checks see internal
                            `**` so no asymmetric strip; matcher False.
"""
from __future__ import annotations

from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


_MULTICHAR_WRAPPERS = ("***", "**", "__", "~~")
_SYMMETRIC_WRAPPERS = ("***", "**", "__", "~~", "`")


def _strip_once(s: str) -> str:
    """One pass: try symmetric, then asymmetric trailing, then asymmetric leading."""
    for w in _SYMMETRIC_WRAPPERS:
        if s.startswith(w) and s.endswith(w) and len(s) > 2 * len(w):
            return s[len(w):-len(w)].strip()
    for w in _MULTICHAR_WRAPPERS:
        if s.endswith(w) and not s.startswith(w):
            inner = s[: -len(w)]
            if w not in inner:
                return inner.strip()
    for w in _MULTICHAR_WRAPPERS:
        if s.startswith(w) and not s.endswith(w):
            inner = s[len(w):]
            if w not in inner:
                return inner.strip()
    return s


def _strip_wrappers(s: str) -> str:
    prev: str | None = None
    while s and s != prev:
        prev = s
        s = _strip_once(s)
    return s


def _matches(ctx: ComponentContext) -> bool:
    answer = (ctx.answer or "").strip()
    if not answer:
        return False
    return _strip_wrappers(answer) != answer


def _handler(ctx: ComponentContext) -> Decision:
    answer = (ctx.answer or "").strip()
    cleaned = _strip_wrappers(answer)
    if not cleaned or cleaned == answer:
        return Decision.allow()
    return Decision.rewrite(cleaned)


COMPONENT = Component(
    name="answer_markdown_strip",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_answer_emit",
    matcher=_matches,
    handler=_handler,
    priority=130,
    emits=(),
    trust=Trust(
        evidence_anchor=(
            "Anchored on three stable contracts outside the evidence traces: "
            "(1) CommonMark / GFM Markdown spec §6.4 defines `**...**`, "
            "`__...__`, `~~...~~`, and `` `...` `` as symmetric inline "
            "emphasis / code delimiters around span content; "
            "(2) agent/base.py::_FINAL_RE — the published extraction regex "
            "`r\"FINAL ANSWER:\\s*(.+?)\\s*$\"` with re.MULTILINE. When the "
            "LLM writes `**FINAL ANSWER: <x>**` the literal `FINAL ANSWER:` "
            "anchor begins AFTER the opening `**`, so the opening wrapper "
            "is excluded but the closing one is captured into group(1) "
            "(lazy `.+?` extended to `\\s*$` end-of-line). The leak shape "
            "is directly derivable from the regex semantics — no induction "
            "from evidence traces required; "
            "(3) agent/base.py::SYSTEM_PROMPT mandates plain-text answers "
            "(`strings: no extra prefix or explanation`), so any Markdown "
            "emphasis in ctx.answer is always wrong by construction against "
            "the GAIA scorer which compares to unwrapped ground truth. "
            "Two observed instances (50ec8903 iter7, 4d0aa727 iter2) "
            "corroborate but are NOT what anchors the hook."
        ),
        blast_radius="local",
        rollback_when=(
            "train-30 accuracy decreases vs the iter7 frontier (24/30) — i.e. "
            "stripping removes characters that belonged to the canonical "
            "answer (e.g. a GAIA answer whose ground truth literally ends "
            "with `**`, which would be unprecedented given the SYSTEM_PROMPT "
            "constraint). The matcher fires only when stripping would change "
            "the answer; the strip is conservative (single-char wrappers "
            "never asymmetric-stripped; multi-char only when the wrapper "
            "does not occur inside the remaining span)."
        ),
        out_of_evidence_probe=(
            "Case A: answer='42' — no wrapper chars; matcher False; no "
            "change. Case B: answer='**42**' — symmetric strip → '42'. "
            "Case C: answer='5 * 7 = 35' — no edge multi-char wrapper; "
            "matcher False. Case D: answer='green, white**' (observed "
            "leak signature) — trailing `**` strip → 'green, white'. "
            "Case E: answer='' — matcher False; no change. "
            "Case F: answer='`code`' — symmetric strip → 'code'. "
            "Case G: answer='foo*' — single-char `*` not in asymmetric "
            "strip set; matcher False; preserves legitimate trailing "
            "asterisks. Case H: answer='a **b** c' — internal `**` blocks "
            "both leading and trailing asymmetric strip; matcher False."
        ),
        fallback=(
            "Matcher returns False when ctx.answer is empty / whitespace "
            "OR when stripping would not change the answer. Handler "
            "returns Decision.allow() when cleaned is empty or unchanged. "
            "A missed fire (matcher false-negative) preserves the iter7 "
            "frontier behaviour exactly — the original (possibly wrapped) "
            "answer is emitted as-is."
        ),
    ),
)
