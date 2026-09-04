from __future__ import annotations

from typing import Protocol


class RateLimiter(Protocol):
    """Token bucket. Adapters: Redis (Lua, atomic), InMemory.

    A non-atomic limiter under concurrency is a suggestion, not a limit - the
    Redis adapter does check-and-decrement inside a single Lua script.
    """

    async def check(self, scope: str, identity: str) -> None:
        """Consume one token, or raise ``RateLimited(retry_after)``."""
        ...


class IdempotencyStore(Protocol):
    """Maps an idempotency key to the result it already produced.

    Without this, a retried ``submit_claim`` files a duplicate insurance claim.
    """

    async def get(self, key: str) -> str | None: ...
    async def put(self, key: str, value: str, ttl_s: int = 86_400) -> None: ...
