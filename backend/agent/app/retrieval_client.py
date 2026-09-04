"""HTTP client for the retrieval service.

Satisfies the ``PolicySearcher`` protocol the policy tool depends on, so tests
substitute an in-memory index without patching anything.
"""

from __future__ import annotations

import httpx

from libs.contracts import Chunk


class RetrievalClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        """Hybrid search.

        Short timeout on purpose - retrieval is local, so it is either fast or
        broken, and a long wait only burns budget the model still needs. Ten
        seconds rather than five because the very first call after a cold start
        pays for connection setup while the service finishes warming; a real
        ReadTimeout on turn one is what raised this.
        """
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/search",
                json={"query": query, "top_k": top_k},
            )
            response.raise_for_status()
            return [Chunk.model_validate(c) for c in response.json()["chunks"]]
