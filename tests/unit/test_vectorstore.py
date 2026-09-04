"""Both ``VectorStore`` adapters, held to the same contract.

The point of a parametrised suite here is that it is the same suite: if the two
adapters do not behave identically the port is a fiction, and swapping backends
would change answers rather than just storage.
"""

from __future__ import annotations

import pytest

from libs.adapters.vectorstore_qdrant import InMemoryVectorStore, QdrantVectorStore
from libs.contracts import Chunk

DIM = 4
A = Chunk(chunk_id="doc::a", text="water damage burst pipe", source_file="sample_policy.md",
          section_id="section-1", section_title="Section 1: Home Water Damage Coverage",
          char_start=0, char_end=20)
B = Chunk(chunk_id="doc::b", text="jewelry appraisal receipts", source_file="sample_policy.md",
          section_id="section-2", section_title="Section 2: Personal Property Protection",
          char_start=21, char_end=50)
VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0]


@pytest.fixture(params=["memory", "qdrant"])
def store(request, tmp_path):
    if request.param == "memory":
        s = InMemoryVectorStore(dimension=DIM)
    else:
        qdrant_client = pytest.importorskip("qdrant_client")  # noqa: F841
        s = QdrantVectorStore(tmp_path / "qdrant", dimension=DIM)
    yield s
    s.close()


async def test_upsert_then_search_returns_the_nearer_chunk(store) -> None:
    await store.upsert([A, B], [VEC_A, VEC_B])
    hits = await store.search(VEC_A, top_k=1)
    assert [h.chunk_id for h in hits] == ["doc::a"]


async def test_hits_carry_a_score(store) -> None:
    await store.upsert([A, B], [VEC_A, VEC_B])
    assert (await store.search(VEC_A, top_k=1))[0].score is not None


async def test_payload_round_trips_everything_a_citation_needs(store) -> None:
    """A search result must be a complete Chunk - a second lookup to build the
    citation would be a chance for the two to disagree."""
    await store.upsert([A], [VEC_A])
    hit = (await store.search(VEC_A, top_k=1))[0]
    assert hit.source_file == "sample_policy.md"
    assert hit.section_title == "Section 1: Home Water Damage Coverage"
    assert hit.citation == "sample_policy.md § Section 1: Home Water Damage Coverage"


async def test_reingest_updates_in_place(store) -> None:
    """`make seed` after editing a policy file must not duplicate sections."""
    await store.upsert([A, B], [VEC_A, VEC_B])
    await store.upsert([A, B], [VEC_A, VEC_B])
    assert len(await store.all_chunks()) == 2


async def test_all_chunks_backs_the_bm25_corpus(store) -> None:
    """Both retrievers must score the same documents, or RRF fuses rankings
    over different corpora."""
    await store.upsert([A, B], [VEC_A, VEC_B])
    assert {c.chunk_id for c in await store.all_chunks()} == {"doc::a", "doc::b"}


async def test_top_k_is_respected(store) -> None:
    await store.upsert([A, B], [VEC_A, VEC_B])
    assert len(await store.search(VEC_A, top_k=1)) == 1
    assert len(await store.search(VEC_A, top_k=5)) == 2


async def test_citation_survives_the_store_round_trip(store) -> None:
    """The payload carries the citation itself, not just the fields to rebuild
    it - so the retrieval layer is the only thing that names a section."""
    await store.upsert([A], [VEC_A])
    hit = (await store.search(VEC_A, top_k=1))[0]
    assert hit.citation == A.citation


async def test_a_stored_citation_is_not_rewritten_on_read(store) -> None:
    """If a document was indexed under an older citation format, reading it
    back must return what was stored - otherwise the index and the answer
    disagree about what the source was called."""
    legacy = A.model_copy(update={"chunk_id": "doc::legacy", "citation": "old::Section 1"})
    await store.upsert([legacy], [VEC_A])
    hits = await store.search(VEC_A, top_k=2)
    assert "old::Section 1" in [h.citation for h in hits]
