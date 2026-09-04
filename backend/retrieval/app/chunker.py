"""Section-aware chunking for policy documents.

The chunk unit is the markdown section, because a citation must name something
a human can go and read. ``sample_policy.md`` has two sections, so chunk ==
section today; ``parent_section`` exists so a long section can be sub-split
later without changing a single citation string.

Pure functions, no I/O. See spec §6.
"""

from __future__ import annotations

import re
from pathlib import Path

from libs.contracts import Chunk
from libs.contracts.retrieval import format_citation

_SECTION_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def chunk_markdown(text: str, source_file: str) -> list[Chunk]:
    """Split a markdown policy document on ``##`` headings.

    Content before the first ``##`` (the ``#`` title line) is dropped: it names
    the document, not a coverage rule, and citing it would be useless.

    Args:
        text: Raw markdown.
        source_file: Basename used in the citation, e.g. ``sample_policy.md``.

    Returns:
        One chunk per section, in document order, each carrying the character
        span it came from so a citation can be traced back to the source.
    """
    matches = list(_SECTION_RE.finditer(text))
    chunks: list[Chunk] = []

    for i, m in enumerate(matches):
        title = m.group("title").strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue

        section_id = _slugify(title.split(":")[0])
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}::{section_id}",
                text=f"{title}\n\n{body}",
                source_file=source_file,
                section_id=section_id,
                section_title=title,
                parent_section=None,
                char_start=m.start(),
                char_end=body_end,
                # Set here, where the section is identified, so the citation is
                # metadata produced once at ingest rather than re-derived by
                # every consumer.
                citation=format_citation(source_file, title),
            )
        )
    return chunks


def chunk_file(path: str | Path) -> list[Chunk]:
    p = Path(path)
    return chunk_markdown(p.read_text(encoding="utf-8"), p.name)
