"""Retry, circuit breaking, and idempotency.

The three work together and none of them is sufficient alone:

  * Retry alone amplifies an outage - every caller hammering a dead provider.
  * A breaker alone gives up on transient blips that a single retry would fix.
  * Both together, without idempotency, cheerfully file the same insurance
    claim twice when a write times out after the server already processed it.

That third one is the failure that matters in this domain, and it is about
twenty lines to prevent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

# Retry these: the request may succeed on a later attempt.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class TransientError(Exception):
    """Something worth retrying. Carries the server's own Retry-After if given."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """A malformed request does not improve on attempt two. Never retried."""


class CircuitOpen(Exception):
    def __init__(self, name: str, reopens_in: float) -> None:
        super().__init__(f"Circuit {name!r} is open for another {reopens_in:.1f}s")
        self.reopens_in = reopens_in


@dataclass
class RetryPolicy:
    """Bounded by attempts *and* a wall-clock deadline, because a human is
    waiting. Voice uses a much tighter deadline than REST: dead air reads as a
    crash long before a slow answer does."""

    attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    deadline_s: float = 20.0

    def backoff(self, attempt: int) -> float:
        """Exponential with *full* jitter.

        Full jitter rather than equal jitter because the failure mode here is
        correlated: several agent replicas trip the same provider 429 at the
        same instant, and retrying in lockstep just reproduces the spike.
        """
        ceiling = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return random.uniform(0, ceiling)


@dataclass
class CircuitBreaker:
    """Fail fast to the fallback provider instead of retrying into a wall."""

    name: str
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            # Half-open: let one request through to test the water.
            self._opened_at = None
            self._failures = self.failure_threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def raise_if_open(self) -> None:
        if self.is_open:
            assert self._opened_at is not None
            remaining = self.reset_after_s - (time.monotonic() - self._opened_at)
            raise CircuitOpen(self.name, remaining)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    breaker: CircuitBreaker | None = None,
) -> T:
    """Run ``fn``, retrying only transient failures within the deadline."""
    started = time.monotonic()
    last: Exception | None = None

    for attempt in range(policy.attempts):
        if breaker is not None:
            breaker.raise_if_open()

        try:
            result = await fn()
        except PermanentError:
            if breaker is not None:
                breaker.record_success()  # the provider answered; we asked wrongly
            raise
        except TransientError as exc:
            last = exc
            if breaker is not None:
                breaker.record_failure()
            delay = exc.retry_after if exc.retry_after is not None else policy.backoff(attempt)
        except Exception as exc:
            last = exc
            if breaker is not None:
                breaker.record_failure()
            delay = policy.backoff(attempt)
        else:
            if breaker is not None:
                breaker.record_success()
            return result

        elapsed = time.monotonic() - started
        if attempt == policy.attempts - 1 or elapsed + delay >= policy.deadline_s:
            break
        await asyncio.sleep(delay)

    raise last or TransientError("exhausted retries")


def idempotency_key(user_id: str, tool: str, args: dict[str, Any], turn: int) -> str:
    """Stable across retries of the same logical write, different for a genuine
    second claim.

    ``turn`` is what distinguishes "the network ate my response, try again"
    from "I really do want to file a second identical claim next week".
    """
    payload = json.dumps({"u": user_id, "t": tool, "a": args, "n": turn}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
