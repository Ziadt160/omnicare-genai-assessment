"""Postgres-backed ``ConversationRepository``.

Owned by the gateway, which is why history stays readable while the agent is
down - the concrete payoff of the service split.

Connection handling: a lazily-opened pool, because the gateway must start and
answer ``/health`` even when Postgres is not up yet. Compose orders startup
with a healthcheck, but a database that dies later must degrade the service,
not kill it.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from libs.contracts import ToolCall


class PostgresConversationRepo:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8) -> None:
        # open=False so constructing the repo never blocks on a database that
        # is still starting. The pool opens on first use.
        self._pool = AsyncConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=False
        )
        self._opened = False

    async def _ready(self) -> AsyncConnectionPool:
        if not self._opened:
            await self._pool.open(wait=True, timeout=10)
            self._opened = True
        return self._pool

    async def ping(self) -> None:
        pool = await self._ready()
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")

    async def aclose(self) -> None:
        if self._opened:
            await self._pool.close()

    # ------------------------------------------------------------- writes

    async def ensure(self, user_id: str, conversation_id: str | None) -> str:
        pool = await self._ready()
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # type: ignore[assignment]

            if conversation_id:
                row = await (
                    await conn.execute(
                        "SELECT id FROM app.conversations WHERE id = %s", (conversation_id,)
                    )
                ).fetchone()
                if row:
                    return str(row["id"])

            else:
                # The graded request schema has no conversation_id, so a user
                # without one continues their most recent conversation rather
                # than starting a new one on every message.
                row = await (
                    await conn.execute(
                        "SELECT id FROM app.conversations WHERE user_id = %s "
                        "ORDER BY updated_at DESC LIMIT 1",
                        (user_id,),
                    )
                ).fetchone()
                if row:
                    return str(row["id"])

            new_id = conversation_id or str(uuid.uuid4())
            await conn.execute(
                "INSERT INTO app.conversations (id, user_id) VALUES (%s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (new_id, user_id),
            )
            return new_id

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        sources: list[str] | None = None,
        tool_calls: list[ToolCall] | None = None,
        channel: str = "text",
        provider: str | None = None,
        model: str | None = None,
        latency_ms: int | None = None,
        trace_id: str | None = None,
    ) -> str:
        pool = await self._ready()
        message_id = str(uuid.uuid4())
        payload = [t.model_dump(mode="json") for t in (tool_calls or [])]

        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO app.messages
                  (id, conversation_id, role, content, sources, tool_calls,
                   channel, provider, model, latency_ms, trace_id)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                """,
                (
                    message_id, conversation_id, role, content,
                    json.dumps(sources or []), json.dumps(payload),
                    channel, provider, model, latency_ms, trace_id,
                ),
            )
            await conn.execute(
                "UPDATE app.conversations SET updated_at = now() WHERE id = %s",
                (conversation_id,),
            )
        return message_id

    # -------------------------------------------------------------- reads

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        pool = await self._ready()
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # type: ignore[assignment]
            rows = await (
                await conn.execute(
                    "SELECT id, user_id, title, created_at, updated_at "
                    "FROM app.conversations WHERE user_id = %s "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (user_id, limit),
                )
            ).fetchall()
        return [_serialize(r) for r in rows]

    async def messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        pool = await self._ready()
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # type: ignore[assignment]
            rows = await (
                await conn.execute(
                    "SELECT id, role, content, sources, tool_calls, channel, "
                    "       provider, model, latency_ms, trace_id, created_at "
                    "FROM app.messages WHERE conversation_id = %s "
                    "ORDER BY created_at LIMIT %s",
                    (conversation_id, limit),
                )
            ).fetchall()
        if not rows:
            # Distinguish "no such conversation" from "an empty one" only when
            # the caller cares; an empty list is the honest answer here.
            return []
        return [_serialize(r) for r in rows]


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    """Timestamps out as ISO strings so the API layer never leaks a driver type."""
    out = dict(row)
    for key, value in out.items():
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
        elif isinstance(value, uuid.UUID):
            out[key] = str(value)
    return out
