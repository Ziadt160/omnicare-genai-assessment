"""Contracts for the retrieval service.

Citations are the reason chunking is section-shaped: a citation must name a
section a human can go and read, so the section is the natural chunk unit.
``parent_section`` lets long sections sub-split later without breaking the
citation string.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchPolicyArgs(BaseModel):
    """Tool arguments for ``search_policy_documents``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="The policyholder's coverage question, rephrased for search.",
        examples=["burst pipe water damage coverage limit deductible"],
    )
    top_k: int = Field(
        default=3, ge=1, le=10,
        description="How many policy sections to retrieve. Default 3.",
    )


CITATION_TEMPLATE = "{source_file} § {section_title}"


def format_citation(source_file: str, section_title: str) -> str:
    """The single place the citation format is defined.

    Everything else reads ``Chunk.citation``. Keeping the format in one
    function means changing it is one edit, and means no consumer has to know
    the convention in order to quote a section.
    """
    return CITATION_TEMPLATE.format(source_file=source_file, section_title=section_title)


class Chunk(BaseModel):
    """A section of a policy document, with the citation carried as metadata.

    ``citation`` is a stored field rather than a derived property. The section
    is metadata, and the citation is the metadata that matters: it travels with
    the chunk into the vector store payload, out through the tool result, and
    into ``ChatResponse.sources`` unchanged. Nothing downstream reconstructs
    it, so nothing downstream can reconstruct it differently - which is exactly
    how a citation drifts from the section it is supposed to name.

    It is populated automatically when omitted, so a caller cannot forget it.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    source_file: str = Field(examples=["sample_policy.md"])
    section_id: str = Field(examples=["section-1"])
    section_title: str = Field(examples=["Section 1: Home Water Damage Coverage"])
    parent_section: str | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    score: float | None = None

    citation: str = Field(
        default="",
        description="The exact string that appears in ChatResponse.sources.",
        examples=["sample_policy.md § Section 1: Home Water Damage Coverage"],
    )

    @model_validator(mode="after")
    def _fill_citation(self) -> "Chunk":
        if not self.citation:
            self.citation = format_citation(self.source_file, self.section_title)
        return self


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[Chunk] = Field(default_factory=list)
    query: str

    @property
    def citations(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.citation, None)
        return list(seen)
