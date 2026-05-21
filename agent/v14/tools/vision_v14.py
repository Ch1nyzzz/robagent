"""v14 vision_describe — routes the multimodal call through the shared
`agent.llm.chat()` so it inherits the rate-slot wiring (QPM + TPM + cooldown).

v8's vision.py called the v9-rewritten `_acquire_rate_slot()` with no
arguments, which surfaced in v15 traces as `vision_error:_acquire_rate_slot()
missing 1 required positional argument: 'token_estimate'` for 10 / 11 image
tasks. Rather than re-derive token estimates locally, we hand the multimodal
message structure straight to `chat()`, which already estimates tokens,
acquires the slot, and handles retries / cooldown uniformly.

The function returns the same source-dict shape as v8 so the workflow
downstream wiring is unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from typing import Any

from agent.events import EventLog
from agent.llm import chat


VISION_MODEL = os.environ.get("VISION_MODEL", "meta-llama/Llama-Vision-Free")
_MAX_DESC_CHARS = 4000
_MAX_TOKENS = 2048


def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


_PROMPT = (
    "You are looking at an image attached to a benchmark task. "
    "Describe the image factually and answer the task if you can. "
    "Report visible numbers, colors, labels, shapes exactly. "
    "Do NOT invent details. If something is unclear, say so."
)


def vision_describe(
    path: str,
    question: str,
    *,
    log: EventLog,
    parent: str,
) -> dict[str, Any]:
    """Ask the vision model to describe the image with respect to the task.

    Returns {ok, content, source_id, url, ...} or {ok=False, reason}.
    """
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="vision_describe",
        args={"path": os.path.basename(path), "question_len": len(question)},
    )
    try:
        data_uri = _data_uri(path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{_PROMPT}\n\nQuestion: {question}"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        resp = chat(messages=messages, model=VISION_MODEL, max_tokens=_MAX_TOKENS)
        content = (resp.get("content") or "")[:_MAX_DESC_CHARS]
        if not content.strip():
            log.emit(
                "tool.returned",
                parent=call_id,
                tool="vision_describe",
                ok=False,
                error="empty description",
            )
            return {"ok": False, "reason": "empty_description"}
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        source_id = f"S_img_{h}"
        log.emit(
            "source.opened",
            parent=call_id,
            source_id=source_id,
            kind="vision",
            url=f"file://{path}",
            content_chars=len(content),
            content_hash=h,
        )
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="vision_describe",
            ok=True,
            source_id=source_id,
            n_chars=len(content),
        )
        return {
            "ok": True,
            "kind": "vision",
            "title": os.path.basename(path),
            "url": f"file://{path}",
            "content": content,
            "source_id": source_id,
            "content_hash": h,
        }
    except Exception as e:
        msg = str(e)
        # Together AI returns 400 "Unable to access non-serverless model ..."
        # when the free vision endpoint is gated. Surface this as a clean
        # `model_capability_gap` block reason so the harness does not retry
        # and the do_not_revisit set keeps it out of subsequent iterations.
        is_cap_gap = ("non-serverless" in msg) or ("Unable to access" in msg and "400" in msg)
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="vision_describe",
            ok=False,
            error=repr(e)[:200],
            capability_gap=is_cap_gap,
        )
        if is_cap_gap:
            return {"ok": False, "reason": "model_capability_gap:vision_unavailable"}
        return {"ok": False, "reason": f"vision_error:{e}"}
