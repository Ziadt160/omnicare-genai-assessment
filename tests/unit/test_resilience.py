"""Retry, circuit breaking and idempotency.

The idempotency tests are the ones that matter for this domain: a retried write
must not file a second insurance claim.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from libs.resilience.policy import (
    CircuitBreaker,
    CircuitOpen,
    PermanentError,
    RetryPolicy,
    TransientError,
    call_with_retry,
    idempotency_key,
)

FAST = RetryPolicy(attempts=3, base_delay_s=0.001, max_delay_s=0.002, deadline_s=5.0)


async def test_succeeds_without_retrying() -> None:
    calls = []

    async def fn() -> str:
        calls.append(1)
        return "ok"

    assert await call_with_retry(fn, FAST) == "ok"
    assert len(calls) == 1


async def test_transient_failure_is_retried_then_succeeds() -> None:
    calls = []

    async def fn() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("429 rate limited")
        return "ok"

    assert await call_with_retry(fn, FAST) == "ok"
    assert len(calls) == 3


async def test_permanent_failure_is_never_retried() -> None:
    """A malformed request does not improve on attempt two - retrying it just
    spends quota to get the same 422."""
    calls = []

    async def fn() -> str:
        calls.append(1)
        raise PermanentError("422 invalid policy_number")

    with pytest.raises(PermanentError):
        await call_with_retry(fn, FAST)
    assert len(calls) == 1


async def test_attempts_are_bounded() -> None:
    calls = []

    async def fn() -> str:
        calls.append(1)
        raise TransientError("still down")

    with pytest.raises(TransientError):
        await call_with_retry(fn, FAST)
    assert len(calls) == FAST.attempts


async def test_server_retry_after_is_honoured_over_our_backoff() -> None:
    """When the provider tells us how long to wait, arguing with it just earns
    another 429."""
    policy = RetryPolicy(attempts=2, base_delay_s=10.0, max_delay_s=10.0, deadline_s=5.0)
    calls = []

    async def fn() -> str:
        calls.append(1)
        raise TransientError("429", retry_after=0.001)

    with pytest.raises(TransientError):
        await call_with_retry(fn, policy)
    assert len(calls) == 2, "our 10s backoff would have blown the 5s deadline"


async def test_breaker_opens_and_fails_fast() -> None:
    """Once open, calls stop reaching the provider at all - that is the point.
    Retrying into a wall is how one outage becomes a thundering herd."""
    breaker = CircuitBreaker("groq", failure_threshold=2, reset_after_s=60)
    calls = []

    async def fn() -> str:
        calls.append(1)
        raise TransientError("down")

    for _ in range(2):
        with pytest.raises(TransientError):
            await call_with_retry(fn, RetryPolicy(attempts=1, deadline_s=1), breaker)

    assert breaker.is_open
    before = len(calls)
    with pytest.raises(CircuitOpen):
        await call_with_retry(fn, RetryPolicy(attempts=1, deadline_s=1), breaker)
    assert len(calls) == before, "an open circuit must not call the provider"


async def test_breaker_half_opens_after_the_reset_window() -> None:
    breaker = CircuitBreaker("groq", failure_threshold=1, reset_after_s=0.0)
    breaker.record_failure()
    assert not breaker.is_open, "a zero-length window should half-open immediately"


async def test_permanent_error_does_not_trip_the_breaker() -> None:
    """The provider answered; we asked wrongly. Opening the circuit on our own
    bad request would take down a healthy dependency."""
    breaker = CircuitBreaker("groq", failure_threshold=1)

    async def fn() -> str:
        raise PermanentError("422")

    with pytest.raises(PermanentError):
        await call_with_retry(fn, RetryPolicy(attempts=1, deadline_s=1), breaker)
    assert not breaker.is_open


# ------------------------------------------------------------- idempotency

CLAIM = {"policy_number": "POL-1092", "claim_type": "Water Damage", "amount": "1200.00"}


def test_same_write_retried_yields_the_same_key() -> None:
    """This is what stops a timed-out submit_claim from filing twice."""
    a = idempotency_key("usr_123", "submit_claim", CLAIM, turn=4)
    b = idempotency_key("usr_123", "submit_claim", dict(reversed(list(CLAIM.items()))), turn=4)
    assert a == b, "key must not depend on dict ordering"


def test_a_genuinely_new_claim_yields_a_different_key() -> None:
    """A policyholder filing an identical claim on a later turn must not be
    silently deduplicated into the first one."""
    assert idempotency_key("usr_123", "submit_claim", CLAIM, turn=4) != idempotency_key(
        "usr_123", "submit_claim", CLAIM, turn=9
    )


def test_different_users_never_collide() -> None:
    assert idempotency_key("usr_a", "submit_claim", CLAIM, 1) != idempotency_key(
        "usr_b", "submit_claim", CLAIM, 1
    )


def test_changed_amount_changes_the_key() -> None:
    other = {**CLAIM, "amount": "1300.00"}
    assert idempotency_key("usr_123", "submit_claim", CLAIM, 1) != idempotency_key(
        "usr_123", "submit_claim", other, 1
    )


def test_backoff_is_bounded_and_jittered() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0)
    samples = [policy.backoff(5) for _ in range(50)]
    assert all(0 <= s <= 4.0 for s in samples)
    assert len(set(samples)) > 1, "full jitter must not be deterministic"


# ------------------------------------------------------- egress token budget

async def test_token_budget_holds_before_the_limit_is_breached() -> None:
    """The limit that actually binds on a free tier is tokens, not requests.

    Groq allows 1000 requests/day but 8000 tokens/minute; at ~2000 tokens a
    call that is four calls a minute, and a ReAct turn is two of them. A
    requests-only limiter sails past it, collects 429s, trips the breaker, and
    every later turn fails fast - which is precisely what the first live eval
    run against Groq did.
    """
    from agent.app.providers.resilient import EgressLimiter

    limiter = EgressLimiter(requests_per_minute=6000, tokens_per_minute=5000)
    limiter._estimate = 2000

    started = asyncio.get_running_loop().time()
    for _ in range(2):
        await limiter.acquire()
        limiter.record(2000)
    assert asyncio.get_running_loop().time() - started < 0.5, "first two must not wait"

    # The third would take spend to 6000 against a 5000 budget.
    assert limiter._spent_last_minute(time.monotonic()) == 4000


async def test_recorded_usage_replaces_the_reservation() -> None:
    """The estimate self-tunes from the provider's own accounting rather than
    drifting on a guess."""
    from agent.app.providers.resilient import EgressLimiter

    limiter = EgressLimiter(requests_per_minute=6000, tokens_per_minute=100_000)
    limiter._estimate = 2000

    await limiter.acquire()
    limiter.record(500)

    assert limiter._spent_last_minute(time.monotonic()) == 500
    assert limiter._estimate < 2000, "estimate should move toward observed usage"


async def test_zero_token_budget_disables_the_check() -> None:
    """Providers without a published TPM limit should not be throttled."""
    from agent.app.providers.resilient import EgressLimiter

    limiter = EgressLimiter(requests_per_minute=6000, tokens_per_minute=0)
    started = asyncio.get_running_loop().time()
    for _ in range(5):
        await limiter.acquire()
        limiter.record(100_000)
    assert asyncio.get_running_loop().time() - started < 0.5


async def test_a_provider_reporting_no_usage_is_not_throttled() -> None:
    """A local or stubbed model reports no token usage. Leaving those
    reservations in the ledger throttled a provider that answers instantly and
    turned the e2e suite into timeouts."""
    from agent.app.providers.resilient import EgressLimiter

    limiter = EgressLimiter(requests_per_minute=6000, tokens_per_minute=5000)
    limiter._estimate = 2000

    started = asyncio.get_running_loop().time()
    for _ in range(6):
        await limiter.acquire()
        limiter.record(0)          # provider reported nothing

    assert limiter._spent_last_minute(time.monotonic()) == 0
    assert asyncio.get_running_loop().time() - started < 0.5


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Error code: 429 - rate limit reached ... on tokens per minute (TPM)", "TransientError"),
        ("Error code: 429 - rate limit reached ... on tokens per day (TPD): Limit 200000", "PermanentError"),
        ("Error code: 429 - rate limit reached ... on requests per day (RPD)", "PermanentError"),
    ],
)
def test_a_daily_quota_is_not_retried(message: str, expected: str) -> None:
    """A per-minute ceiling is worth waiting out inside a request; a daily one
    is not. Groq publishes RPM and TPM in headers but enforces tokens-per-day
    only in the 429 body, so the limiter cannot see it coming - retrying just
    burns the attempts and the deadline for an error that will not clear.
    """
    from agent.app.providers.resilient import classify

    exc = Exception(message)
    exc.status_code = 429  # type: ignore[attr-defined]
    assert type(classify(exc)).__name__ == expected
