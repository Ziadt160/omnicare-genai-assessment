"""The retrieval service.

Separate from the agent for one concrete reason: it holds the embedding model
in RAM with a slow warm start. In-process, every agent restart would reload it,
and agent restarts are the common case because the agent is the tier that talks
to flaky external providers.

The index is built at startup, which is why the compose healthcheck for this
service has a generous start_period - the first `docker compose up` would
otherwise race the warm-up and look broken.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from libs.contracts import Chunk, HealthResponse, SearchPolicyArgs
from .index import FastEmbedder, HybridIndex
from .settings import RetrievalSettings


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[Chunk] = Field(default_factory=list)
    query: str
    citations: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks_indexed: int
    sections: list[str] = Field(default_factory=list)


settings = RetrievalSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = FastEmbedder(settings.embedding_model)
    store = None
    if settings.vector_backend == "qdrant":
        from libs.adapters.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(settings.qdrant_path, dimension=embedder.dimension)

    index = HybridIndex(embedder, store)
    await index.ingest_directory(settings.source_dir)
    app.state.index = index
    yield


app = FastAPI(title="OmniCare Retrieval", version="1.0.0", lifespan=lifespan)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Healthy only once the index is warm.

    Reporting healthy while the embedding model is still loading would let the
    agent start and immediately fail its first coverage question.
    """
    index: HybridIndex | None = getattr(app.state, "index", None)
    return HealthResponse(status="healthy" if index and index.ready else "degraded")


@app.post("/api/v1/search", response_model=SearchResponse, tags=["search"])
async def search(request: SearchPolicyArgs) -> SearchResponse:
    """Hybrid search over the policy documents.

    Dense embeddings plus BM25, fused with Reciprocal Rank Fusion. BM25 earns
    its place on exact tokens - $25,000, POL-1092, "deductible" - which dense
    vectors blur.
    """
    index: HybridIndex = app.state.index
    chunks = await index.search(request.query, request.top_k)
    return SearchResponse(
        chunks=chunks,
        query=request.query,
        citations=list(dict.fromkeys(c.citation for c in chunks)),
    )


@app.get("/api/v1/sections", tags=["search"])
async def sections() -> list[dict[str, str]]:
    """Every indexed section, in full.

    Search answers "what is relevant to this question"; this answers "what does
    the policy say", which is a different question and the one a rule check
    needs. A coverage cap must be read from the section that governs the claim,
    not from whichever section an embedder ranked first - it puts Personal
    Property above Home Water Damage for "what is my deductible?", and a cap
    taken from the wrong section would refuse a valid claim while quoting a
    figure that does not govern it.
    """
    index: HybridIndex = app.state.index
    return [
        {"section_title": c.section_title, "text": c.text, "source_file": c.source_file}
        for c in index.chunks
    ]


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["admin"])
async def ingest() -> IngestResponse:
    """Re-read the source directory. Called by `make seed` after editing a
    policy document, so the index does not need a container restart."""
    index: HybridIndex = app.state.index
    count = await index.ingest_directory(settings.source_dir)
    return IngestResponse(
        chunks_indexed=count, sections=[c.section_title for c in index.chunks]
    )
