from __future__ import annotations

from typing import Protocol

from libs.contracts import ToolCall


class ConversationRepository(Protocol):
    """Durable chat history. Adapters: Postgres, InMemory.

    Owned by the gateway, so history stays readable while the agent is down.
    """

    async def ensure(self, user_id: str, conversation_id: str | None) -> str:
        """Return an existing conversation id, or create one and return it."""
        ...

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
        ...

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        ...

    async def messages(self, conversation_id: str, limit: int = 200) -> list[dict]:
        ...
