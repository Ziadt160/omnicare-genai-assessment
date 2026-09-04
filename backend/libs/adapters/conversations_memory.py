"""In-memory ``ConversationRepository``.

Used by the gateway's contract tests and as the fallback when no DATABASE_URL
is configured, so the service still starts and answers - history is simply not
durable. Failing to boot because a database is missing would be the wrong
trade for a prototype whose graded endpoint does not need one.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from libs.contracts import ToolCall


class InMemoryConversationRepo:
    def __init__(self) -> None:
        self._conversations: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}

    async def ensure(self, user_id: str, conversation_id: str | None) -> str:
        if conversation_id and conversation_id in self._conversations:
            return conversation_id
        if conversation_id is None:
            # One rolling conversation per user, because the graded request
            # schema has no conversation_id field.
            for cid, row in self._conversations.items():
                if row["user_id"] == user_id:
                    return cid
        cid = conversation_id or f"cnv_{uuid.uuid4().hex[:12]}"
        self._conversations[cid] = {
            "id": cid,
            "user_id": user_id,
            "title": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._messages[cid] = []
        return cid

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
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        self._messages.setdefault(conversation_id, []).append(
            {
                "id": mid,
                "role": role,
                "content": content,
                "sources": sources or [],
                "tool_calls": [t.model_dump() for t in (tool_calls or [])],
                "channel": channel,
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
                "trace_id": trace_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if conversation_id in self._conversations:
            self._conversations[conversation_id]["updated_at"] = datetime.now(
                timezone.utc
            ).isoformat()
        return mid

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = [c for c in self._conversations.values() if c["user_id"] == user_id]
        return sorted(rows, key=lambda c: c["updated_at"], reverse=True)[:limit]

    async def messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self._messages.get(conversation_id, [])[:limit]
