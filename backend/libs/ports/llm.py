from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol


class LLMProvider(Protocol):
    """Chat completion with tool calling.

    Groq, NVIDIA NIM, GitHub Models and Ollama are all OpenAI-compatible, so
    one ``OpenAICompatibleProvider`` covers every hosted option; only
    ``base_url`` and ``model`` differ. ``FakeLLM`` implements this with
    scripted tool-call sequences and is what makes TDD possible - see §15.
    """

    name: str
    model: str

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        ...

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        ...
