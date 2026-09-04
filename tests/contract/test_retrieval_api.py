"""The retrieval service, with a deterministic stub embedder.

The point of injecting the embedder is exactly this: the whole hybrid path -
dense ranking, BM25, RRF, citation assembly - is exercised without loading an
ONNX model, so these run in milliseconds in CI.

The stub embeds on term overlap rather than returning noise, so a dense
regression still shows up here rather than being masked by BM25 carrying the
result on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from libs.contracts import Chunk
from retrieval.app.chunker import chunk_file
from retrieval.app.hybrid import tokenize
from retrieval.app.index import HybridIndex
from retrieval.app.main import app

POLICY = Path(__file__).parents[2] / "data" / "sample_policy.md"
SECTION_1 = "sample_policy.md § Section 1: Home Water Damage Coverage"
SECTION_2 = "sample_policy.md § Section 2: Personal Property Protection"

VOCAB = [
    "water", "damage", "pipe", "burst", "sudden", "deductible", "flood",
    "gradual", "leak", "excluded", "$25,000", "$500",
    "electronics", "furniture", "jewelry", "appraisal", "receipts",
    "$10,000", "$2,500", "property", "covered",
]


class StubEmbedder:
    """Bag-of-words over a fixed vocabulary. Deterministic and term-sensitive,
    so cosine similarity behaves like a real embedder for these queries."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            terms = set(tokenize(text))
            out.append([1.0 if term in terms else 0.0 for term in VOCAB])
        return out

    @property
    def dimension(self) -> int:
        return len(VOCAB)


@pytest_asyncio.fixture()
async def client() -> AsyncIterator[httpx.AsyncClient]:
    index = HybridIndex(StubEmbedder())
    await index.ingest(chunk_file(POLICY))
    app.state.index = index

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_health_is_healthy_once_the_index_is_warm(client) -> None:
    r = await client.get("/api/v1/health")
    assert r.json() == {"status": "healthy"}


async def test_health_is_degraded_before_ingest() -> None:
    """Reporting healthy while the model is still loading would let the agent
    start and immediately fail its first coverage question."""
    app.state.index = HybridIndex(StubEmbedder())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/api/v1/health")).json()["status"] == "degraded"


@pytest.mark.parametrize(
    ("query", "expected_citation"),
    [
        ("burst pipe water damage", SECTION_1),
        ("what is my deductible", SECTION_1),
        ("is flood damage covered", SECTION_1),
        ("jewelry appraisal receipts", SECTION_2),
        ("electronics and furniture limit", SECTION_2),
    ],
)
async def test_search_returns_the_right_section_first(client, query, expected_citation) -> None:
    r = await client.post("/api/v1/search", json={"query": query, "top_k": 3})
    assert r.status_code == 200
    assert r.json()["citations"][0] == expected_citation


async def test_recall_at_3_is_total(client) -> None:
    """Two sections, top_k=3 - anything less than both is an ingest bug."""
    r = await client.post("/api/v1/search", json={"query": "coverage", "top_k": 3})
    assert set(r.json()["citations"]) == {SECTION_1, SECTION_2}


async def test_chunks_carry_everything_a_citation_needs(client) -> None:
    r = await client.post("/api/v1/search", json={"query": "burst pipe", "top_k": 1})
    chunk = Chunk.model_validate(r.json()["chunks"][0])
    assert chunk.source_file == "sample_policy.md"
    assert chunk.section_title.startswith("Section 1")
    assert chunk.citation == SECTION_1
    assert chunk.score is not None


async def test_the_exclusion_travels_with_the_coverage(client) -> None:
    """If the exclusion were split from the limit, retrieval could hand the
    model "$25,000" without "flood damage is excluded" - the most dangerous
    wrong answer this system can produce."""
    r = await client.post("/api/v1/search", json={"query": "flood", "top_k": 1})
    text = r.json()["chunks"][0]["text"]
    assert "excluded" in text
    assert "$25,000" in text


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({}, "missing query"),
        ({"query": ""}, "empty query"),
        ({"query": "x", "top_k": 0}, "top_k below 1"),
        ({"query": "x", "top_k": 99}, "top_k above 10"),
        ({"query": "x", "unknown": 1}, "unknown key"),
    ],
)
async def test_invalid_search_requests_are_rejected(client, payload, why) -> None:
    assert (await client.post("/api/v1/search", json=payload)).status_code == 422, why


async def test_ingest_is_idempotent(client) -> None:
    """`make seed` after editing a policy file must not duplicate sections."""
    first = (await client.post("/api/v1/ingest")).json()
    second = (await client.post("/api/v1/ingest")).json()
    assert first["chunks_indexed"] == second["chunks_indexed"]
