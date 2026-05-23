"""tau2 candidate mh_tau2_iter1_baseline_kb_dedup.

Hypothesis
----------
The baseline LLMAgent thrashes KB_search: when it cannot find a detail it
re-issues the same question with slightly different wording. KB_search returns
only the single top-ranked document, so each re-wording returns a document the
agent has already seen — it never assembles the reference data it needs and
then computes wrong values / recommends the wrong product, and the wasted
turns crowd out the rest of the task.

Mechanism observed (v0 banking sims)
------------------------------------
task_001 issued 21 KB_search calls, task_024 12, task_018 14, task_055 16,
task_057 23 — overwhelmingly re-wordings of one question — and all failed.

Change
------
A deterministic guard (`kb_guard.KBSearchGuard`) flags a KB_search as
redundant when its query introduces no content token absent from the union of
every KB_search already issued this conversation. Such a query cannot surface
new information. When the agent emits one, the turn is regenerated with an
explicit note telling it to use what it has, search a genuinely new topic, or
proceed. A query that introduces any new term is always allowed, so the guard
can never hide an un-searched topic. The system prompt also gains general
guidance on how KB_search behaves. The LLM still makes every genuine decision.
"""
from __future__ import annotations

from typing import Optional

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
)
from tau2.utils.llm_utils import generate

from agent_tau2.mh_tau2_iter1_baseline_kb_dedup.kb_guard import KBSearchGuard

# General, domain-agnostic guidance appended to the system prompt.
RETRIEVAL_GUIDANCE = """
<retrieval_guidance>
The KB_search tool returns only the SINGLE highest-ranked knowledge-base
document for your query. Re-issuing a query that is only a reworded version of
one you already tried returns that same document and wastes a turn — it does
not surface additional content.

To research efficiently:
- Issue one focused KB_search per distinct subject (one per card, account
  class, fee type, or policy) instead of rephrasing the same question.
- Before searching again, check whether a document you already retrieved
  already answers the question.
- If repeated searches are not surfacing a needed detail, proceed with the
  information you have or ask the user — do not search indefinitely.
- Base every calculation and recommendation on the documents actually
  retrieved; never guess at rates, fees, or policy values.
</retrieval_guidance>
""".strip()

# Maximum number of generations per turn (1 original + retries).
MAX_GENERATIONS = 3

KB_SEARCH_TOOL = "KB_search"


def _tool_call_query(tool_call) -> Optional[str]:
    """Return the `query` argument of a KB_search tool call, or None."""
    if getattr(tool_call, "name", None) != KB_SEARCH_TOOL:
        return None
    args = getattr(tool_call, "arguments", None)
    if isinstance(args, dict):
        q = args.get("query")
        return q if isinstance(q, str) else None
    if isinstance(args, str):
        # Some providers hand back arguments as a JSON string.
        try:
            import json

            parsed = json.loads(args)
            q = parsed.get("query") if isinstance(parsed, dict) else None
            return q if isinstance(q, str) else None
        except Exception:
            return None
    return None


def _kb_queries(message: AssistantMessage) -> list[str]:
    """All KB_search query strings in an assistant message."""
    out: list[str] = []
    for tc in message.tool_calls or []:
        q = _tool_call_query(tc)
        if q is not None:
            out.append(q)
    return out


class KBDedupAgent(LLMAgent):
    """LLMAgent that blocks and re-prompts redundant KB_search re-wordings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = KBSearchGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + RETRIEVAL_GUIDANCE

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._guard.reset()
        return super().get_init_state(message_history)

    def _first_redundant_query(self, message: AssistantMessage) -> Optional[str]:
        """Return the first redundant KB_search query in the message, if any."""
        for q in _kb_queries(message):
            if self._guard.is_redundant(q):
                return q
        return None

    def _generate(self, state: LLMAgentState, guidance: Optional[str]) -> AssistantMessage:
        """Run one LLM generation, optionally with a transient turn note."""
        system_messages = state.system_messages
        if guidance:
            base = state.system_messages[0].content
            system_messages = [
                SystemMessage(
                    role="system",
                    content=f"{base}\n\n<turn_note>\n{guidance}\n</turn_note>",
                )
            ]
        messages = system_messages + state.messages
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

    def generate_next_message(self, message, state: LLMAgentState):
        """Respond to a user or tool message, suppressing redundant KB_search."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for _ in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            redundant_q = self._first_redundant_query(assistant_message)
            if redundant_q is None:
                break
            # Regenerate with a note steering the agent off the dead-end search.
            guidance = self._guard.guidance(redundant_q)

        # Record whatever KB_search queries the accepted turn actually issued.
        for q in _kb_queries(assistant_message):
            self._guard.register(q)

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return KBDedupAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
