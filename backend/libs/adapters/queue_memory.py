"""In-memory ``JobQueue``. Lets the gateway's contract tests run the full
enqueue-and-await path with no Redis container.

Deliberately mirrors the Redis Streams semantics that matter: a job stream with
a pending depth, per-run event streams, and a terminal event that releases the
subscriber. If a test passes here and fails against Redis, the difference is
real behaviour rather than a shortcut in the fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from libs.contracts import RunEvent


class InMemoryQueue:
    def __init__(self) -> None:
        self.jobs: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._streams: dict[str, list[RunEvent]] = {}
        self._wakeups: dict[str, asyncio.Event] = {}
        self._pending = 0

    async def enqueue(self, run_id: str, job: dict[str, Any], deadline_s: float) -> None:
        self._pending += 1
        self._streams.setdefault(run_id, [])
        self._wakeups.setdefault(run_id, asyncio.Event())
        await self.jobs.put({"run_id": run_id, "deadline_s": deadline_s, **job})

    async def depth(self) -> int:
        return self._pending

    async def publish(self, event: RunEvent) -> None:
        self._streams.setdefault(event.run_id, []).append(event)
        self._wakeups.setdefault(event.run_id, asyncio.Event()).set()
        if event.is_terminal:
            self._pending = max(0, self._pending - 1)

    async def subscribe(  # type: ignore[override]
        self, run_id: str, timeout_s: float
    ) -> AsyncIterator[RunEvent]:
        cursor = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        while True:
            events = self._streams.setdefault(run_id, [])
            while cursor < len(events):
                event = events[cursor]
                cursor += 1
                yield event
                if event.is_terminal:
                    return

            remaining = deadline - loop.time()
            if remaining <= 0:
                return

            wakeup = self._wakeups.setdefault(run_id, asyncio.Event())
            wakeup.clear()
            try:
                await asyncio.wait_for(wakeup.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return


class InMemoryRateLimiter:
    """Always allows. Rate limiting has its own tests; the gateway's contract
    tests should not be coupled to a window."""

    async def check(self, scope: str, identity: str) -> None:
        return None


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def put(self, key: str, value: str, ttl_s: int = 86_400) -> None:
        self._values[key] = value
