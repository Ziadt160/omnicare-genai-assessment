"""``mock_claims.json`` as the claims system of record.

This is the default adapter because the assessment names the file explicitly.
It is written to be genuinely safe under ``--scale agent=4``, which is the
whole reason the claims store did not need to become its own service:

  * ``asyncio.Lock``  serialises writers inside one process
  * ``filelock``      serialises writers across containers - the lock is taken
                      on a shared bind mount, so every replica contends for the
                      same inode
  * write-temp + ``os.replace``  makes the swap atomic, so a reader never
                      observes a half-written file even mid-write

See spec §10 and docs/adr/0005. ``PostgresClaimsRepo`` implements the same
port and is selected by ``CLAIMS_BACKEND=postgres``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from filelock import FileLock

from libs.contracts import Claim, SubmitClaimArgs
from libs.errors import ClaimNotFound

LOCK_TIMEOUT_S = 5.0


class JsonFileClaimsRepo:
    """File-backed ``ClaimsRepository``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._file_lock = FileLock(str(self.path) + ".lock", timeout=LOCK_TIMEOUT_S)

    # ------------------------------------------------------------------ read

    def _read(self) -> list[Claim]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [Claim.model_validate(c) for c in raw]

    async def get(self, claim_id: str) -> Claim:
        claims = await asyncio.to_thread(self._read)
        for c in claims:
            if c.claim_id == claim_id:
                return c
        raise ClaimNotFound(claim_id, known_ids=[c.claim_id for c in claims])

    async def list_ids(self) -> list[str]:
        claims = await asyncio.to_thread(self._read)
        return [c.claim_id for c in claims]

    # ----------------------------------------------------------------- write

    @staticmethod
    def _next_id(claims: list[Claim]) -> str:
        highest = max((int(c.claim_id.split("-")[1]) for c in claims), default=8820)
        return f"CLM-{highest + 1:04d}"

    def _append_sync(self, args: SubmitClaimArgs) -> Claim:
        with self._file_lock:
            claims = self._read()
            claim = Claim(
                claim_id=self._next_id(claims),
                status="Submitted",
                **args.model_dump(),
            )
            payload = [c.model_dump(mode="json") for c in [*claims, claim]]
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)  # atomic on POSIX and Windows
            return claim

    async def append(self, args: SubmitClaimArgs) -> Claim:
        """Append one claim and return it with its assigned confirmation id."""
        async with self._lock:
            return await asyncio.to_thread(self._append_sync, args)
