"""The hybrid index.

Dense and sparse are kept side by side over the same chunk list and fused on
rank, so neither retriever needs to know about the other. The embedder is
injected rather than constructed here, which is what lets the whole index run
in tests with a deterministic stub instead of loading an ONNX model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from libs.contracts import Chunk
from .chunker import chunk_file
from .hybrid import BM25, fuse


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 through fastembed.

    fastembed is ONNX, so this pulls no PyTorch. That single choice is the
    difference between a ~400 MB retrieval image and a ~2.5 GB one, which in
    turn is the difference between a two-minute first run and a twenty-minute
    one. Imported lazily so importing this module stays cheap.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dimension = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    @property
    def dimension(self) -> int:
        return self._dimension


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class HybridIndex:
    """Dense + BM25 + RRF over one corpus.

    The dense side goes through the ``VectorStore`` port, so swapping the
    in-memory adapter for Qdrant is one environment variable and changes
    nothing else. BM25 always builds its corpus from ``store.all_chunks()`` -
    both retrievers must score the same documents, or RRF is fusing rankings
    over different corpora.

    Async because the port is: a real vector database is I/O, and pretending
    otherwise would mean blocking the event loop on every query.
    """

    def __init__(self, embedder: Embedder, store: Any | None = None) -> None:
        from libs.adapters.vectorstore_qdrant import InMemoryVectorStore

        self.embedder = embedder
        self.store = store or InMemoryVectorStore(dimension=embedder.dimension)
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None

    async def ingest_directory(self, directory: str | Path, pattern: str = "*.md") -> int:
        chunks: list[Chunk] = []
        for path in sorted(Path(directory).glob(pattern)):
            chunks.extend(chunk_file(path))
        return await self.ingest(chunks)

    async def ingest(self, chunks: list[Chunk]) -> int:
        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            await self.store.upsert(chunks, vectors)
        self.chunks = await self.store.all_chunks()
        self.bm25 = BM25(self.chunks) if self.chunks else None
        return len(self.chunks)

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.bm25 is not None

    async def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        if not self.ready:
            return []

        # Over-fetch on both sides: RRF needs enough depth to reward agreement,
        # and a document ranked 4th by one retriever and 1st by the other
        # should still surface.
        depth = max(top_k * 2, 5)
        query_vector = self.embedder.embed([query])[0]

        dense = await self.store.search(query_vector, depth)
        dense_ids = [c.chunk_id for c in dense]

        assert self.bm25 is not None
        sparse_ids = self.bm25.rank(query, top_k=depth)

        by_id = {c.chunk_id: c for c in self.chunks}
        return fuse(dense_ids, sparse_ids, by_id, top_k)
