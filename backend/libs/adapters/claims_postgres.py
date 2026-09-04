"""Postgres-backed ``ClaimsRepository``.

Selected with ``AGENT_CLAIMS_BACKEND=postgres``. The JSON adapter stays the
default because the brief names ``mock_claims.json`` explicitly, but this is
the whole point of the port: swapping the claims store is one environment
variable and no change anywhere else - not to the tools, not to the graph, not
to the API.

Two things the file adapter cannot do, and the reason this exists beyond making
the port honest:

* ``append`` is a single ``INSERT ... RETURNING`` inside the database, so claim
  ids are allocated by a sequence rather than by reading the file, finding the
  highest id and adding one. Under concurrency the file version needs a lock to
  be correct; this one is correct because the database serialises it.
* Claims survive ``docker compose down -v`` alongside conversations and
  checkpoints, in one backup rather than a volume plus a file.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from libs.contracts import Claim, SubmitClaimArgs
from libs.errors import ClaimNotFound

# Claim ids continue the series in the supplied fixture (CLM-8821, CLM-9014),
# so a store seeded from that file and one created empty produce ids of the
# same shape.
FIRST_CLAIM_NUMBER = 9015

SCHEMA = """
CREATE SCHEMA IF NOT EXISTS app;

CREATE SEQUENCE IF NOT EXISTS app.claim_number_seq START WITH {start};

CREATE TABLE IF NOT EXISTS app.claims (
  claim_id       TEXT PRIMARY KEY,
  policy_number  TEXT        NOT NULL,
  claim_type     TEXT        NOT NULL,
  status         TEXT        NOT NULL,
  amount         NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
  description    TEXT        NOT NULL DEFAULT '',
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS claims_policy_idx ON app.claims (policy_number);
""".format(start=FIRST_CLAIM_NUMBER)


class PostgresClaimsRepo:
    """Same port as ``JsonFileClaimsRepo``; different durability story."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 4) -> None:
        # open=False so constructing the repo never blocks on a database that
        # is still starting; the pool opens on first use.
        self._pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False)
        self._ready_done = False

    async def _ready(self) -> AsyncConnectionPool:
        if not self._ready_done:
            await self._pool.open(wait=True, timeout=10)
            async with self._pool.connection() as conn:
                await conn.execute(SCHEMA)
            self._ready_done = True
        return self._pool

    async def aclose(self) -> None:
        if self._ready_done:
            await self._pool.close()

    # ------------------------------------------------------------- reads

    async def get(self, claim_id: str) -> Claim:
        pool = await self._ready()
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # type: ignore[assignment]
            row = await (
                await conn.execute(
                    "SELECT claim_id, policy_number, claim_type, status, amount, "
                    "description FROM app.claims WHERE claim_id = %s",
                    (claim_id,),
                )
            ).fetchone()

            if row is None:
                known = await (
                    await conn.execute("SELECT claim_id FROM app.claims ORDER BY claim_id")
                ).fetchall()
                raise ClaimNotFound(claim_id, known_ids=[r["claim_id"] for r in known])

        return Claim.model_validate(_as_domain(row))

    async def list_ids(self) -> list[str]:
        pool = await self._ready()
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # type: ignore[assignment]
            rows = await (
                await conn.execute("SELECT claim_id FROM app.claims ORDER BY claim_id")
            ).fetchall()
        return [r["claim_id"] for r in rows]

    # ------------------------------------------------------------- writes

    async def append(self, args: SubmitClaimArgs) -> Claim:
        """One statement, so the id is allocated by the database.

        The file adapter has to read every claim, find the highest id and add
        one, which is only safe because it holds a lock. Here the sequence does
        it, and two agent replicas filing simultaneously cannot collide.
        """
        pool = await self._ready()
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # type: ignore[assignment]
            row = await (
                await conn.execute(
                    """
                    INSERT INTO app.claims
                      (claim_id, policy_number, claim_type, status, amount, description)
                    VALUES (
                      'CLM-' || lpad(nextval('app.claim_number_seq')::text, 4, '0'),
                      %s, %s, 'Submitted', %s, %s
                    )
                    RETURNING claim_id, policy_number, claim_type, status, amount, description
                    """,
                    (args.policy_number, args.claim_type, args.amount, args.description),
                )
            ).fetchone()

        assert row is not None
        return Claim.model_validate(_as_domain(row))

    async def seed(self, claims: list[Claim]) -> int:
        """Load the supplied fixture on first start. Never overwrites."""
        pool = await self._ready()
        inserted = 0
        async with pool.connection() as conn:
            for claim in claims:
                result = await conn.execute(
                    """
                    INSERT INTO app.claims
                      (claim_id, policy_number, claim_type, status, amount, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (claim_id) DO NOTHING
                    """,
                    (claim.claim_id, claim.policy_number, claim.claim_type,
                     claim.status, claim.amount, claim.description or ""),
                )
                inserted += result.rowcount or 0
        return inserted


def _as_domain(row: dict[str, Any]) -> dict[str, Any]:
    """Driver row to domain shape - no driver types escape the adapter."""
    out = dict(row)
    if isinstance(out.get("amount"), Decimal):
        out["amount"] = out["amount"]
    out["description"] = out.get("description") or ""
    return out
