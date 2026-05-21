from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# DeepSeek official API (OpenAI-compatible). Models: deepseek-v4-pro | deepseek-v4-flash.
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "deepseek-v4-pro")
_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not _API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY not set in environment or .env")

_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)

# DeepSeek's official API does not hard-gate by qpm/tpm the way Together did;
# it throttles dynamically with 429s, which the retry loop already handles.
# Keep the limiter as a soft safety net set very high (effectively off).
_RATE_LIMIT_PER_MIN = int(os.environ.get("LLM_RATE_LIMIT_PER_MIN", "5000"))
_TOKEN_LIMIT_PER_MIN = int(os.environ.get("LLM_TOKEN_LIMIT_PER_MIN", "50000000"))
# Total wall-time budget per logical chat() invocation (acquire + call + waits)
_MAX_WALL_SECONDS = float(os.environ.get("LLM_MAX_WALL_SECONDS", "480"))
# Per-attempt API timeout when contacting the upstream service
_PER_CALL_TIMEOUT = float(os.environ.get("LLM_PER_CALL_TIMEOUT", "150"))
# Max number of API attempts for transient errors (each gated by the queue)
_MAX_API_ATTEMPTS = int(os.environ.get("LLM_MAX_API_ATTEMPTS", "10"))

# Shared cooldown when the provider returns 429. All threads wait until this
# moment before attempting another call; reset opportunistically.
_cooldown_until: float = 0.0


def _set_cooldown(seconds: float) -> None:
    global _cooldown_until
    until = time.monotonic() + max(0.5, seconds)
    if until > _cooldown_until:
        _cooldown_until = until


def _wait_for_cooldown(deadline: float) -> None:
    # Rate-limit cooldown is a QUEUE, not a failure. It is bounded (<=90s per
    # _set_cooldown), so wait it out fully — never fail a call because the
    # provider asked everyone to pause. `deadline` is intentionally ignored
    # here; it only governs the actual API call + genuine-error retries.
    while True:
        remaining = _cooldown_until - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0))

_rate_lock = threading.Lock()
_rate_window: "deque[float]" = deque()
_token_window: "deque[tuple[float, int]]" = deque()


def _prune_windows(now: float) -> None:
    while _rate_window and now - _rate_window[0] > 60.0:
        _rate_window.popleft()
    while _token_window and now - _token_window[0][0] > 60.0:
        _token_window.popleft()


def _tokens_in_window() -> int:
    return sum(n for _, n in _token_window)


def _acquire_rate_slot(token_estimate: int, *, deadline: float) -> None:
    """Block until QPM and TPM budgets both allow this call.

    Strict pre-acquisition: the slot is reserved BEFORE the API call. This is a
    QUEUE — a slot always frees within the 60s window, so wait it out fully
    rather than failing the call. `deadline` is intentionally ignored; it only
    governs the actual API call + genuine-error retries.
    """
    while True:
        with _rate_lock:
            now = time.monotonic()
            _prune_windows(now)
            qpm_ok = len(_rate_window) < _RATE_LIMIT_PER_MIN
            tpm_ok = _tokens_in_window() + token_estimate <= _TOKEN_LIMIT_PER_MIN
            if qpm_ok and tpm_ok:
                _rate_window.append(now)
                _token_window.append((now, token_estimate))
                return
            waits: list[float] = []
            if not qpm_ok and _rate_window:
                waits.append(60.0 - (now - _rate_window[0]) + 0.05)
            if not tpm_ok and _token_window:
                waits.append(60.0 - (now - _token_window[0][0]) + 0.05)
            wait_for = max(0.1, min(waits) if waits else 0.5)
        time.sleep(wait_for)


def _record_actual_token_usage(actual_tokens: int) -> None:
    """Refine the most recent token window slot with actual usage from response."""
    with _rate_lock:
        if _token_window:
            ts, _ = _token_window[-1]
            _token_window[-1] = (ts, actual_tokens)


def _estimate_token_cost(messages: list[dict[str, Any]], max_tokens: int) -> int:
    # rough: prompt tokens ~ chars/4 + max_tokens for completion budget
    char_count = sum(len(str(m.get("content", ""))) for m in messages)
    return int(char_count / 4) + max_tokens


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Identify provider rate-limit responses by class name or message."""
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


def _is_transient_error(exc: BaseException) -> bool:
    """Network / timeout / 5xx — worth re-attempting after a backoff."""
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "connection", "apierror", "internalserver", "service")):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("timeout", "timed out", "connection", "502", "503", "504"))


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call the upstream chat model with strict rate-limit gating.

    Contract (v11+):
    - Acquires a QPM+TPM slot BEFORE every API attempt.
    - Retries only on transient/rate-limit errors, with bounded attempts and
      jittered backoff. Each retry re-acquires a fresh slot.
    - Never raises a tenacity RetryError wrapper — surfaces the underlying
      exception so callers / traces see the real cause.
    """
    estimate = _estimate_token_cost(messages, max_tokens)
    start = time.monotonic()
    deadline = start + _MAX_WALL_SECONDS
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_API_ATTEMPTS + 1):
        _wait_for_cooldown(deadline)
        _acquire_rate_slot(token_estimate=estimate, deadline=deadline)
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=_PER_CALL_TIMEOUT,
            )
        except TypeError:
            # Older SDKs may not accept `timeout=` kwarg; retry without it.
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BaseException as e:
            last_exc = e
            if attempt >= _MAX_API_ATTEMPTS:
                raise
            is_rl = _is_rate_limit_error(e)
            if not (_is_transient_error(e) or is_rl):
                raise
            if is_rl:
                # Provider 429 — set a shared cooldown so every thread pauses.
                # 15s base + jitter; doubles per consecutive 429 from this thread.
                cooldown = min(15.0 * (2 ** (attempt - 1)), 90.0)
                cooldown *= 0.8 + 0.4 * random.random()
                _set_cooldown(cooldown)
            backoff = min(2.0 * (2 ** (attempt - 1)), 30.0)
            backoff *= 0.5 + random.random()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(backoff, max(0.0, remaining)))
            continue
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        actual = getattr(usage, "total_tokens", None) if usage else None
        if actual:
            _record_actual_token_usage(int(actual))
        return {
            "model": model,
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            },
        }
    # Defensive — loop should have returned or raised.
    if last_exc:
        raise last_exc
    raise RuntimeError("chat() exhausted attempts without exception")
