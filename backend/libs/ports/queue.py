from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from libs.contracts import RunEvent


class JobQueue(Protocol):
    """Work queue plus per-run event stream.

    Adapters: RedisStreams (consumer groups distribute across agent replicas),
    InMemory. See §9 - the queue and the token transport are the same primitive.
    """

    async def enqueue(self, run_id: str, job: dict[str, Any], deadline_s: float) -> None:
        ...

    async def depth(self) -> int:
        """Pending entries. Backs admission control - see ``QueueSaturated``."""
        ...

    async def publish(self, event: RunEvent) -> None:
        ...

    def subscribe(self, run_id: str, timeout_s: float) -> AsyncIterator[RunEvent]:
        """Tail ``stream:{run_id}``; resumable by entry id after a dropped socket."""
        ...
