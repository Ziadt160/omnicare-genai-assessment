"""The LLM call path: egress rate limit, retry, circuit breaker, fallback.

Everything here was written and tested in ``libs.resilience`` and then called
from nowhere. This module is the connection - it wraps a chat model so the
graph's agent node gets all four behaviours without knowing about any of them.

Order matters and is not arbitrary:

  1. **Egress limiter first.** Waiting 200 ms is cheaper than spending a
     request to be told 429. This is the piece that stops an eval run of thirty
     cases from tripping a free-tier RPM ceiling in the first two seconds.
  2. **Retry inside the breaker.** A blip deserves a second attempt; an outage
     does not deserve thirty replicas each making three.
  3. **Fallback last.** Only once the primary's circuit is open, so a single
     slow response does not flap between providers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from libs.resilience.policy import (
    CircuitOpen,
    PermanentError,
    RetryPolicy,
    TransientError,
    call_with_retry,
)
from .registry import make_breaker

log = logging.getLogger("omnicare.agent.llm")

# Substrings that mean "the provider answered, we asked wrongly". Retrying
# these spends quota to receive the identical error.
_PERMANENT_HINTS = (
    "invalid_api_key", "authentication", "unauthorized", "forbidden",
    "model_not_found", "does not exist", "invalid_request_error",
    "context_length", "maximum context",
)


def classify(exc: Exception) -> Exception:
    """Map a provider exception onto the retry taxonomy.

    Providers report failures through wildly different exception types, so
    classification is by status code and message rather than by class.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    text = str(exc).lower()

    if status in (400, 401, 403, 404, 422):
        return PermanentError(str(exc))
    if any(hint in text for hint in _PERMANENT_HINTS):
        return PermanentError(str(exc))

    if status == 429 or "rate limit" in text or "too many requests" in text:
        # A *daily* budget is not something to wait out inside a request. Groq
        # publishes RPM and TPM in response headers but enforces a tokens-per-day
        # ceiling as well (200k on the free tier) that appears only in the 429
        # body - so the limiter, which reasons in 60s windows, has no way to see
        # it coming and retries into a wall. Treat it as permanent for this run:
        # fail fast with the provider's own message, and let the fallback
        # provider or a human decide.
        if "per day" in text or "tpd" in text or "rpd" in text:
            return PermanentError(str(exc))
        retry_after = getattr(exc, "retry_after", None)
        return TransientError(str(exc), retry_after=retry_after)
    if status is not None and 500 <= int(status) < 600:
        return TransientError(str(exc))
    if any(w in text for w in ("timeout", "timed out", "connection", "temporarily")):
        return TransientError(str(exc))

    # Unknown failures are treated as transient: one wasted retry is cheaper
    # than a dropped answer, and the breaker bounds the damage.
    return TransientError(str(exc))


class EgressLimiter:
    """Client-side ceiling on *both* requests and tokens, applied before the call.

    The counterpart to the gateway's ingress limiter and a genuinely different
    problem: this one protects us from the *provider's* limits, so the correct
    response to hitting it is to wait, not to reject.

    Tokens matter more than requests, and that is not obvious until it bites.
    Groq's free tier allows 1000 requests per day but only **8000 tokens per
    minute**; with a system prompt plus three tool schemas each call is roughly
    2000 tokens, so the real ceiling is about four calls a minute - and a
    ReAct turn is two of them. A requests-only limiter set to 25/min sails past
    that, collects 429s, trips the circuit breaker, and every subsequent turn
    fails fast. That is exactly what the first live eval run against Groq did:
    28/29 locally, 3/29 hosted, with nothing wrong in the graph.

    The window is a rolling 60 s ledger of actual spend, corrected after each
    call with the usage the provider reports, so the estimate self-tunes rather
    than drifting.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int = 0) -> None:
        self.min_interval = 60.0 / max(requests_per_minute, 1)
        self.tokens_per_minute = tokens_per_minute
        self._lock = asyncio.Lock()
        self._last = 0.0
        # (spent_at, tokens) for the trailing minute.
        self._ledger: list[tuple[float, int]] = []
        # Seeded from the first real usage report; 2000 is a reasonable guess
        # for this prompt and is corrected within one call.
        self._estimate = 2000

    def _spent_last_minute(self, now: float) -> int:
        self._ledger = [(t, n) for t, n in self._ledger if now - t < 60.0]
        return sum(n for _, n in self._ledger)

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()

            if self.tokens_per_minute > 0:
                spent = self._spent_last_minute(now)
                if spent + self._estimate > self.tokens_per_minute and self._ledger:
                    # Wait for the oldest entry to age out of the window.
                    wait = max(0.0, 60.0 - (now - self._ledger[0][0])) + 0.25
                    log.info(
                        "egress token budget reached (%d/%d in the last minute); "
                        "holding %.1fs",
                        spent, self.tokens_per_minute, wait,
                    )
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                    self._spent_last_minute(now)

            wait = self.min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
            # Reserve the estimate now; `record` corrects it with real usage.
            self._ledger.append((self._last, self._estimate))

    def record(self, tokens: int) -> None:
        """Replace the reserved estimate with what the call actually cost.

        A provider that reports no usage (a local model, a stub) would leave
        every reservation uncorrected, so the ledger fills with guesses and
        throttles a model that costs nothing. Drop the reservation instead.
        """
        if not self._ledger:
            return
        if tokens <= 0:
            self._ledger.pop()
            return
        stamp, _reserved = self._ledger[-1]
        self._ledger[-1] = (stamp, tokens)
        # Smooth, so one unusually long turn does not skew every later estimate.
        self._estimate = int(0.7 * self._estimate + 0.3 * tokens)


class ResilientChatModel:
    """Wraps one or two chat models with the full call policy.

    Presents just enough of the ``BaseChatModel`` surface for the graph -
    ``bind_tools`` and ``ainvoke`` - rather than subclassing it, because
    inheriting would drag in sync paths and callback machinery this never uses.
    """

    def __init__(
        self,
        primary: BaseChatModel,
        *,
        primary_name: str = "primary",
        fallback: BaseChatModel | None = None,
        fallback_name: str = "fallback",
        policy: RetryPolicy | None = None,
        requests_per_minute: int = 25,
        tokens_per_minute: int = 0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.policy = policy or RetryPolicy()
        self.limiter = EgressLimiter(requests_per_minute, tokens_per_minute)
        self.breaker = make_breaker(primary_name)
        self.active = primary_name
        self._tools: Sequence[Any] = ()

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ResilientChatModel":
        self._tools = tools
        self.primary = self.primary.bind_tools(tools, **kwargs)  # type: ignore[assignment]
        if self.fallback is not None:
            self.fallback = self.fallback.bind_tools(tools, **kwargs)  # type: ignore[assignment]
        return self

    async def _call(self, model: Any, messages: list[BaseMessage]) -> BaseMessage:
        await self.limiter.acquire()
        try:
            result = await model.ainvoke(messages)
        except Exception as exc:
            raise classify(exc) from exc

        # Correct the reserved estimate with the provider's own accounting.
        usage = (getattr(result, "response_metadata", None) or {}).get("token_usage") or {}
        self.limiter.record(int(usage.get("total_tokens") or 0))
        return result

    async def ainvoke(self, messages: list[BaseMessage], **_: Any) -> BaseMessage:
        try:
            result = await call_with_retry(
                lambda: self._call(self.primary, messages), self.policy, self.breaker
            )
            self.active = self.primary_name
            return result
        except (CircuitOpen, TransientError) as exc:
            if self.fallback is None:
                raise
            log.warning(
                "primary provider %s unavailable (%s); falling back to %s",
                self.primary_name, type(exc).__name__, self.fallback_name,
            )
            result = await call_with_retry(
                lambda: self._call(self.fallback, messages), self.policy
            )
            self.active = self.fallback_name
            return result
