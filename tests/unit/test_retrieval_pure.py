"""Chunking, tokenization and rank fusion - all pure, all fast, no I/O."""

from pathlib import Path

import pytest

from retrieval.app.chunker import chunk_markdown
from retrieval.app.hybrid import (
    BM25,
    STOPWORDS,
    analyze,
    fuse,
    reciprocal_rank_fusion,
    tokenize,
)

POLICY = Path(__file__).parents[2] / "data" / "sample_policy.md"


@pytest.fixture(scope="module")
def chunks():
    return chunk_markdown(POLICY.read_text(encoding="utf-8"), "sample_policy.md")


def test_chunks_one_per_section(chunks) -> None:
    assert len(chunks) == 2
    assert [c.section_id for c in chunks] == ["section-1", "section-2"]


def test_document_title_is_not_a_chunk(chunks) -> None:
    """The `#` line names the document, not a coverage rule; citing it is useless."""
    assert not any("General Insurance Policy" == c.section_title for c in chunks)


def test_citation_format_is_stable(chunks) -> None:
    """This exact string appears in ChatResponse.sources and in every eval row."""
    assert chunks[0].citation == (
        "sample_policy.md § Section 1: Home Water Damage Coverage"
    )


def test_chunk_spans_point_back_into_the_source(chunks) -> None:
    text = POLICY.read_text(encoding="utf-8")
    for c in chunks:
        assert c.char_start < c.char_end <= len(text)
        assert c.section_title in text[c.char_start:c.char_end]


def test_exclusions_stay_in_the_same_chunk_as_the_coverage(chunks) -> None:
    """If the exclusion were split from the limit, retrieval could return
    "covered up to $25,000" without "flood damage is excluded" - which is the
    single most dangerous wrong answer this system can give."""
    s1 = chunks[0].text
    assert "$25,000" in s1
    assert "excluded" in s1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("covered up to $25,000", ["covered", "up", "to", "$25,000"]),
        ("policy POL-1092", ["policy", "pol-1092"]),
    ],
)
def test_tokenizer_preserves_money_and_identifiers(text, expected) -> None:
    """Splitting on $ , - would destroy exactly the advantage BM25 has here."""
    assert tokenize(text) == expected


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("burst pipe deductible", "section-1"),
        ("flood damage excluded", "section-1"),
        ("jewelry appraisal receipts", "section-2"),
        ("electronics furniture limit", "section-2"),
    ],
)
def test_bm25_routes_to_the_right_section(chunks, query, expected_section) -> None:
    top = BM25(chunks).rank(query, top_k=1)
    assert top and top[0].endswith(expected_section)


def test_rrf_rewards_agreement_between_retrievers() -> None:
    """A doc ranked by both retrievers must outrank one ranked by only one."""
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert scores["b"] > scores["a"] > scores["c"]


def test_fuse_returns_chunks_with_scores(chunks) -> None:
    by_id = {c.chunk_id: c for c in chunks}
    dense = [chunks[1].chunk_id, chunks[0].chunk_id]
    sparse = [chunks[0].chunk_id]
    out = fuse(dense, sparse, by_id, top_k=2)
    assert [c.section_id for c in out] == ["section-1", "section-2"]
    assert all(c.score is not None for c in out)


# ------------------------------------------------------------- the analyzer

@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("bursts", "burst"),         # the policy says "bursts"; users say "burst"
        ("leaks", "leak"),
        ("receipts", "receipt"),
        ("damages", "damage"),
        ("caused", "cause"),         # classic Porter gets this pair wrong
        ("flooding", "flood"),
        ("excluded", "exclude"),
        ("exceeding", "exceeds"),    # needs the "eed" guard
    ],
)
def test_stemming_is_symmetric_for_domain_vocabulary(a, b) -> None:
    """A lexical matcher needs the same transformation on both sides, not a
    linguistically correct one. These are the pairs that actually occur
    between this policy document and a policyholder's phrasing."""
    assert analyze(a) == analyze(b)


def test_known_stemming_limit_is_documented_not_hidden() -> None:
    """The "eed" guard that fixes exceeds/exceeding breaks agreed/agrees.
    Neither word appears in the policy, so the trade is worth it - but pinning
    it means the next person meets it as a documented limit rather than a
    mystery."""
    assert analyze("agreed") != analyze("agrees")


def test_money_and_identifiers_survive_the_analyzer() -> None:
    """Only alphabetic tokens are stemmed, which is what preserves exactly the
    advantage BM25 has over dense retrieval on this corpus."""
    assert analyze("covered up to $25,000 on POL-1092") == ["cover", "$25,000", "pol-1092"]


def test_stopwords_are_dropped() -> None:
    assert "a" in STOPWORDS and "the" in STOPWORDS
    assert analyze("is the damage covered") == ["damag", "cover"]


def test_an_all_stopword_query_keeps_its_terms() -> None:
    """Matching badly beats matching nothing - an empty term list would make
    BM25 silently return no candidates at all."""
    assert analyze("what is it") != []


def test_stopwords_stop_bm25_inventing_a_match(chunks) -> None:
    """The EV-06 regression. "I have a $4,000 ring" shares only the stopword
    "a" with Section 1, and on a two-document corpus that was enough to pull
    the wrong section to rank 1 - IDF cannot damp a term present in one of two
    documents. Returning nothing is correct here: "ring" is not in the policy,
    which says "jewelry", and bridging that is the dense retriever's job."""
    assert BM25(chunks).rank("I have a $4,000 ring. Do I need anything special?", 2) == []


# ------------------------------------------------- citation as metadata

def test_citation_is_stored_metadata_not_a_derived_property(chunks) -> None:
    """The section is metadata, and the citation is the metadata that matters.

    Stored rather than derived, so it travels with the chunk into the vector
    store payload, out through the tool result, and into ChatResponse.sources
    unchanged. Nothing downstream reconstructs it, so nothing downstream can
    reconstruct it differently.
    """
    dumped = chunks[0].model_dump()
    assert "citation" in dumped
    assert dumped["citation"] == (
        "sample_policy.md § Section 1: Home Water Damage Coverage"
    )


def test_citation_is_filled_when_omitted() -> None:
    """A caller cannot forget it - an empty citation would silently produce an
    answer that cites nothing."""
    from libs.contracts import Chunk

    c = Chunk(
        chunk_id="x", text="t", source_file="policy.md", section_id="s1",
        section_title="Section 9: Flood", char_start=0, char_end=1,
    )
    assert c.citation == "policy.md § Section 9: Flood"


def test_an_explicit_citation_is_preserved() -> None:
    """A chunk read back from a store must keep the citation the store held,
    not have it silently rewritten by the current format."""
    from libs.contracts import Chunk

    c = Chunk(
        chunk_id="x", text="t", source_file="policy.md", section_id="s1",
        section_title="Section 9: Flood", char_start=0, char_end=1,
        citation="legacy-format::Section 9",
    )
    assert c.citation == "legacy-format::Section 9"


def test_format_citation_is_the_only_definition() -> None:
    from libs.contracts.retrieval import format_citation

    assert format_citation("a.md", "Section 1: X") == "a.md § Section 1: X"
