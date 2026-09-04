"""Ingest entrypoint: ``python -m retrieval.app.ingest`` (used by ``make seed``).

Re-reads the source directory and rebuilds the index. Separate from the service
so a policy document can be re-indexed without restarting the container and
paying the embedding-model warm-up again.
"""

from __future__ import annotations

import sys

from .index import FastEmbedder, HybridIndex
from .settings import RetrievalSettings


def main() -> int:
    settings = RetrievalSettings()
    index = HybridIndex(FastEmbedder(settings.embedding_model))
    count = index.ingest_directory(settings.source_dir)

    if not count:
        print(f"No markdown found under {settings.source_dir}", file=sys.stderr)
        return 1

    print(f"Indexed {count} section(s) from {settings.source_dir}:")
    for chunk in index.chunks:
        print(f"  {chunk.citation}  ({chunk.char_end - chunk.char_start} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
