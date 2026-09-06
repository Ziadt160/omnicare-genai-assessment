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
        self._sections: list[tuple[str, str]] | None = None

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

    async def sections(self) -> list[tuple[str, str, str]]:
        """Every section of the policy, as ``(title, text, source_file)``.

        The source file was being dropped here, and the cost only showed up
        once a settlement had to carry a citation: the endpoint has always
        returned it, so an answer grounded in Section 1 was arriving with no
        way to name Section 1 and the UI showed no citation at all.

        Fetched rather than searched: a coverage cap has to come from the
        section that governs the claim, and search answers a different question.
        Cached for the life of the worker - the policy is a document that
        changes when someone edits it and re-ingests, not per request - and a
        failure returns nothing rather than raising, because a rule check that
        cannot read the policy must fall back to not checking, never to
        refusing a claim it could not verify.
        """
        if self._sections is None:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.get(f"{self.base_url}/api/v1/sections")
                    response.raise_for_status()
                    self._sections = [
                        (s["section_title"], s["text"], s.get("source_file", ""))
                        for s in response.json()
                    ]
            except Exception:
                return []
        return self._sections
