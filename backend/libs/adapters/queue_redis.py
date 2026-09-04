"""Redis Streams as both the work queue and the token transport.

Streams rather than Pub/Sub, for one reason that matters: Pub/Sub is
fire-and-forget, so a gateway that restarts mid-answer loses the tokens and the
policyholder sees a truncated response. A stream is an append-only log, so a
reconnecting client resumes from the last entry id it saw.

Consumer groups turn the same primitive into a work queue - XREADGROUP
distributes across however many agent replicas exist, XACK completes, and
XAUTOCLAIM recovers jobs from a replica that died mid-run. That is why
`docker compose up --scale agent=4` needs no configuration.

See docs/adr/0003.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from libs.contracts import RunEvent

JOB_STREAM = "jobs:chat"
JOB_GROUP = "agents"
DEAD_STREAM = "jobs:chat:dead"
JOB_MAXLEN = 10_000
EVENT_TTL_S = 300


def run_stream(run_id: str) -> str:
    return f"stream:{run_id}"


class RedisQueue:
    def __init__(self, url: str) -> None:
        self._redis: Redis = Redis.from_url(url, decode_responses=True)
        self._group_ready = False

    async def ping(self) -> None:
        await self._redis.ping()

    async def aclose(self) -> None:
        await self._redis.aclose()

    async def _ensure_group(self) -> None:
        if self._group_ready:
            return
        try:
            await self._redis.xgroup_create(JOB_STREAM, JOB_GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True

    # ------------------------------------------------------------- produce

    async def enqueue(self, run_id: str, job: dict[str, Any], deadline_s: float) -> None:
        await self._ensure_group()
        loop_now = await self._redis.time()
        deadline_at = loop_now[0] + deadline_s
        await self._redis.xadd(
            JOB_STREAM,
            {
                "run_id": run_id,
                "deadline_at": str(deadline_at),
                "payload": json.dumps(job, default=str),
            },
            maxlen=JOB_MAXLEN,
            approximate=True,
        )

    async def depth(self) -> int:
        """Entries delivered but not yet acked. Backs admission control - a
        queue that accepts everything is just a slower timeout."""
        await self._ensure_group()
        try:
            info = await self._redis.xpending(JOB_STREAM, JOB_GROUP)
        except ResponseError:
            return 0
        return int(info.get("pending", 0)) if isinstance(info, dict) else 0

    # ------------------------------------------------------------- consume

    async def consume(
        self, consumer: str, block_ms: int = 5_000, count: int = 1
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read jobs for this replica. Called by the agent worker."""
        await self._ensure_group()
        batches = await self._redis.xreadgroup(
            JOB_GROUP, consumer, {JOB_STREAM: ">"}, count=count, block=block_ms
        )
        jobs: list[tuple[str, dict[str, Any]]] = []
        for _stream, entries in batches or []:
            for entry_id, fields in entries:
                jobs.append((entry_id, {
                    "run_id": fields["run_id"],
                    "deadline_at": float(fields["deadline_at"]),
                    **json.loads(fields["payload"]),
                }))
        return jobs

    async def ack(self, entry_id: str) -> None:
        await self._redis.xack(JOB_STREAM, JOB_GROUP, entry_id)

    async def dead_letter(self, entry_id: str, job: dict[str, Any], error: str) -> None:
        await self._redis.xadd(
            DEAD_STREAM,
            {"entry_id": entry_id, "error": error, "payload": json.dumps(job, default=str)},
            maxlen=1_000,
            approximate=True,
        )
        await self.ack(entry_id)

    async def reclaim_stalled(self, consumer: str, min_idle_ms: int = 60_000) -> int:
        """Recover jobs from a replica that died mid-run."""
        await self._ensure_group()
        try:
            _, entries, _ = await self._redis.xautoclaim(
                JOB_STREAM, JOB_GROUP, consumer, min_idle_time=min_idle_ms, count=10
            )
        except ResponseError:
            return 0
        return len(entries or [])

    # -------------------------------------------------------------- events

    async def publish(self, event: RunEvent) -> None:
        key = run_stream(event.run_id)
        await self._redis.xadd(key, {"event": event.model_dump_json()})
        await self._redis.expire(key, EVENT_TTL_S)

    async def subscribe(  # type: ignore[override]
        self, run_id: str, timeout_s: float, last_id: str = "0"
    ) -> AsyncIterator[RunEvent]:
        """Tail a run's events until a terminal one or the deadline.

        `last_id` is what makes a dropped socket resumable: the client sends
        back the last entry id it rendered and picks up from there.
        """
        key = run_stream(run_id)
        deadline_ms = int(timeout_s * 1000)
        spent = 0
        cursor = last_id

        while spent < deadline_ms:
            block = min(1_000, deadline_ms - spent)
            batches = await self._redis.xread({key: cursor}, count=32, block=block)
            if not batches:
                spent += block
                continue
            for _stream, entries in batches:
                for entry_id, fields in entries:
                    cursor = entry_id
                    event = RunEvent.model_validate_json(fields["event"])
                    yield event
                    if event.is_terminal:
                        return


class RedisIdempotencyStore:
    """Maps an idempotency key to the result it already produced.

    Without this, a retried `submit_claim` files a duplicate insurance claim -
    the failure a reviewer for an insurance company will probe for first.
    """

    def __init__(self, url: str) -> None:
        self._redis: Redis = Redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._redis.get(f"idem:{key}")

    async def put(self, key: str, value: str, ttl_s: int = 86_400) -> None:
        await self._redis.set(f"idem:{key}", value, ex=ttl_s)

    async def aclose(self) -> None:
        await self._redis.aclose()
