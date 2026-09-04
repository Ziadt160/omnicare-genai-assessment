from __future__ import annotations

from typing import Protocol

from libs.contracts import Claim, SubmitClaimArgs


class ClaimsRepository(Protocol):
    """Storage for insurance claims.

    Adapters: ``JsonFileClaimsRepo`` (default - the assessment names
    ``mock_claims.json`` explicitly), ``PostgresClaimsRepo``, ``InMemoryClaimsRepo``.
    """

    async def get(self, claim_id: str) -> Claim:
        """Return the claim, or raise ``ClaimNotFound``."""
        ...

    async def append(self, args: SubmitClaimArgs) -> Claim:
        """Atomically append a new claim and return it with its assigned id."""
        ...

    async def list_ids(self) -> list[str]:
        """All known claim ids. Backs fuzzy recovery when STT mangles an id."""
        ...
