"""Qdrant-backed ``VectorStore``.

Selected with ``RETRIEVAL_VECTOR_BACKEND=qdrant``. The in-memory adapter stays
the default, and the reason is worth stating rather than hiding: with a
two-section policy document, a vector database is ceremony. Both sections fit
inside `top_k=3`, so nothing it offers - ANN indexing, payload filtering,
persistence across restarts - changes an answer here.

It exists because the port should have a real second implementation. A ports
layer with one adapter each is an assertion; one with two is a demonstration,
and the JSON-to-Postgres claims swap is the same argument. Point the corpus at
a real policy library and this becomes the default.

Local mode (a path, no server) rather than a container: the brief requires a
local vector store, and an extra service for two paragraphs would be the
over-engineering the rest of the design avoids.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from libs.contracts import Chunk

COLLECTION = "policy_sections"


class QdrantVectorStore:
    """Dense index over policy chunks.

    The payload carries the citation itself, not just the fields to rebuild it,
    so a search result is a complete ``Chunk`` and the retrieval layer is the
    only thing that ever decides how a section is named. Re-ingesting a
    document refreshes the stored citations, which is what `make seed` is for.
    """

    def __init__(self, path: str | Path, dimension: int = 384) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        self._dimension = dimension

        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------- writes

    def _upsert_sync(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                # Deterministic id from the chunk id, so re-ingesting the same
                # document updates in place instead of duplicating sections.
                id=abs(hash(chunk.chunk_id)) % (2**63),
                vector=vector,
                payload=chunk.model_dump(mode="json", exclude={"score"}),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=COLLECTION, points=points)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        # qdrant-client is synchronous in local mode; a thread keeps the event
        # loop free during ingest.
        await asyncio.to_thread(self._upsert_sync, chunks, vectors)

    # -------------------------------------------------------------- reads

    def _search_sync(self, vector: list[float], top_k: int) -> list[Chunk]:
        hits = self._client.query_points(
            collection_name=COLLECTION, query=vector, limit=top_k, with_payload=True
        ).points
        return [
            Chunk.model_validate({**hit.payload, "score": hit.score})
            for hit in hits
            if hit.payload
        ]

    async def search(self, vector: list[float], top_k: int) -> list[Chunk]:
        return await asyncio.to_thread(self._search_sync, vector, top_k)

    def _all_sync(self) -> list[Chunk]:
        records, _ = self._client.scroll(
            collection_name=COLLECTION, limit=10_000, with_payload=True
        )
        return [Chunk.model_validate(r.payload) for r in records if r.payload]

    async def all_chunks(self) -> list[Chunk]:
        """Every indexed chunk. BM25 builds its corpus from this, so the two
        retrievers always score the same documents."""
        return await asyncio.to_thread(self._all_sync)

    def close(self) -> None:
        self._client.close()


class InMemoryVectorStore:
    """The default. Holds vectors in a list and scores with cosine.

    Adequate and honest for this corpus, and it is what the unit tests use so
    they need no database.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        by_id = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.chunk_id in by_id:
                index = by_id[chunk.chunk_id]
                self._chunks[index], self._vectors[index] = chunk, vector
            else:
                self._chunks.append(chunk)
                self._vectors.append(vector)

    async def search(self, vector: list[float], top_k: int) -> list[Chunk]:
        from retrieval.app.index import cosine

        scored = sorted(
            (
                (cosine(vector, v), c)
                for v, c in zip(self._vectors, self._chunks, strict=True)
            ),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )
        return [c.model_copy(update={"score": s}) for s, c in scored[:top_k]]

    async def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def close(self) -> None:
        return None
