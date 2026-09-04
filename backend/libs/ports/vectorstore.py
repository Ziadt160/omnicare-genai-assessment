from __future__ import annotations

from typing import Protocol

from libs.contracts import Chunk


class VectorStore(Protocol):
    """Dense vector index. Adapters: Qdrant (local mode), InMemory."""

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    async def search(self, vector: list[float], top_k: int) -> list[Chunk]:
        """Dense search. Returned chunks carry ``score``."""
        ...

    async def all_chunks(self) -> list[Chunk]:
        """Every indexed chunk - the BM25 side builds its corpus from this."""
        ...
