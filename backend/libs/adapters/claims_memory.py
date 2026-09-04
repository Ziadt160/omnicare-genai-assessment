"""In-memory ``ClaimsRepository`` - the adapter that makes unit tests instant."""

from __future__ import annotations

import asyncio

from libs.contracts import Claim, SubmitClaimArgs
from libs.errors import ClaimNotFound


class InMemoryClaimsRepo:
    def __init__(self, seed: list[Claim] | None = None) -> None:
        self._claims: list[Claim] = list(seed or [])
        self._lock = asyncio.Lock()

    async def get(self, claim_id: str) -> Claim:
        for c in self._claims:
            if c.claim_id == claim_id:
                return c
        raise ClaimNotFound(claim_id, known_ids=[c.claim_id for c in self._claims])

    async def list_ids(self) -> list[str]:
        return [c.claim_id for c in self._claims]

    async def append(self, args: SubmitClaimArgs) -> Claim:
        async with self._lock:
            highest = max(
                (int(c.claim_id.split("-")[1]) for c in self._claims), default=8820
            )
            claim = Claim(
                claim_id=f"CLM-{highest + 1:04d}",
                status="Submitted",
                **args.model_dump(),
            )
            self._claims.append(claim)
            return claim
