"""Vision tool — uses Together AI's Llama vision model to describe an image.

Returns a structured source dict suitable for the answer_with_evidence node.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from typing import Any

from agent.events import EventLog
from agent.llm import _client, _acquire_rate_slot  # reuse rate limiter


VISION_MODEL = os.environ.get(
    "VISION_MODEL", "meta-llama/Llama-Vision-Free"
)
_MAX_DESC_CHARS = 4000


def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def vision_describe(
    path: str,
    question: str,
    *,
    log: EventLog,
    parent: str,
) -> dict[str, Any]:
    """Ask the vision model to describe the image with respect to the question.

    Returns {ok, content, source_id, url} or {ok=False, reason}.
    """
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="vision_describe",
        args={"path": os.path.basename(path), "question_len": len(question)},
    )
    try:
        data_uri = _data_uri(path)
        prompt = (
            "You are looking at an image attached to a benchmark question. "
            "Describe the image factually and answer the question if you can. "
            "Report visible numbers, colors, labels, shapes exactly. "
            "Do NOT invent details. If something is unclear, say so."
        )
        _acquire_rate_slot()
        resp = _client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n\nQuestion: {question}"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "")[:_MAX_DESC_CHARS]
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
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="vision_describe",
            ok=False,
            error=repr(e)[:200],
        )
        return {"ok": False, "reason": f"vision_error:{e}"}
