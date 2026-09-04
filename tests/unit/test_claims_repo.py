"""Claims repository behaviour, including the guarantee that justifies keeping
``mock_claims.json`` as the store instead of promoting claims to a service.

The concurrency test is the load-bearing one: the moment ``agent`` scales to N
replicas, N processes append to one file. If this test does not hold, the
scaling story in the README is false.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from libs.adapters.claims_json import JsonFileClaimsRepo
from libs.contracts import Claim, SubmitClaimArgs
from libs.errors import ClaimNotFound

SEED = [
    {"claim_id": "CLM-8821", "policy_number": "POL-1092",
     "claim_type": "Water Damage", "status": "Approved", "amount": 3500.00},
    {"claim_id": "CLM-9014", "policy_number": "POL-3341",
     "claim_type": "Personal Property", "status": "Under Review", "amount": 1200.00},
]


@pytest.fixture()
def repo(tmp_path: Path) -> JsonFileClaimsRepo:
    p = tmp_path / "mock_claims.json"
    p.write_text(json.dumps(SEED, indent=2), encoding="utf-8")
    return JsonFileClaimsRepo(p)


def _args(amount: str = "1200.00") -> SubmitClaimArgs:
    return SubmitClaimArgs(
        policy_number="POL-1092",
        claim_type="Water Damage",
        amount=Decimal(amount),
        description="The washing machine hose burst and flooded the utility room.",
    )


async def test_get_returns_seeded_claim(repo: JsonFileClaimsRepo) -> None:
    claim = await repo.get("CLM-8821")
    assert claim.status == "Approved"
    assert claim.amount == Decimal("3500.00")


async def test_get_unknown_raises_domain_error_with_recovery_hints(
    repo: JsonFileClaimsRepo,
) -> None:
    """The known-ids list is what lets the agent offer "did you mean CLM-8821?"
    instead of a dead-end "not found" - see spec §7."""
    with pytest.raises(ClaimNotFound) as exc:
        await repo.get("CLM-0000")
    assert exc.value.known_ids == ["CLM-8821", "CLM-9014"]


async def test_append_assigns_confirmation_id_and_submitted_status(
    repo: JsonFileClaimsRepo,
) -> None:
    claim = await repo.append(_args())
    assert claim.claim_id == "CLM-9015"
    assert claim.status == "Submitted"
    assert await repo.get("CLM-9015") == claim


async def test_append_preserves_json_number_shape(repo: JsonFileClaimsRepo) -> None:
    """``amount`` must stay a JSON number. Decimal serializes to a string by
    default, which would silently change the shape of the supplied file."""
    await repo.append(_args("1200.00"))
    raw = json.loads(repo.path.read_text(encoding="utf-8"))
    assert all(isinstance(row["amount"], (int, float)) for row in raw)
    assert raw[0]["amount"] == 3500.00


async def test_concurrent_appends_do_not_corrupt_or_drop(
    repo: JsonFileClaimsRepo,
) -> None:
    """Twenty concurrent writers: every claim survives, ids are unique, and the
    file parses. This is the test that makes ``--scale agent=N`` honest."""
    results = await asyncio.gather(*(repo.append(_args()) for _ in range(20)))

    raw = json.loads(repo.path.read_text(encoding="utf-8"))
    assert len(raw) == len(SEED) + 20

    ids = [c.claim_id for c in results]
    assert len(set(ids)) == 20, "duplicate confirmation ids handed to users"
    assert all(Claim.model_validate(row) for row in raw)


async def test_no_temp_file_left_behind(repo: JsonFileClaimsRepo) -> None:
    await repo.append(_args())
    leftovers = list(repo.path.parent.glob("*.tmp"))
    assert leftovers == []


# ------------------------------------------------- examples are prompt text

def test_no_tool_example_is_a_real_record() -> None:
    """A tool's JSON schema is prompt text, and a model copies what it is shown.

    Observed live: asked to file a theft claim, qwen2.5 announced "Policy
    Number: POL-1092" without the policyholder ever saying it. POL-1092 does
    not appear in the system prompt at all - it was the `examples` value on
    submit_claim's own schema. The same schema offered CLM-8821 and 1200.00,
    and all three are live rows in mock_claims.json, so a copied example does
    not merely fabricate a value: it files against a real policyholder's
    policy.

    Format belongs in the pattern and the description, which cannot be pasted
    into a tool call as a value.
    """
    import json
    from pathlib import Path

    from libs.contracts.claims import GetClaimStatusArgs, SubmitClaimArgs

    seeded = json.loads(
        (Path(__file__).resolve().parents[2] / "data" / "mock_claims.json")
        .read_text(encoding="utf-8")
    )
    live = {str(v) for row in seeded for v in row.values()}
    live |= {f"{float(row['amount']):.2f}" for row in seeded}

    offenders = []
    for model in (SubmitClaimArgs, GetClaimStatusArgs):
        for field, spec in model.model_json_schema()["properties"].items():
            for example in spec.get("examples", []):
                if str(example) in live:
                    offenders.append(f"{model.__name__}.{field} = {example!r}")

    assert not offenders, f"tool examples are real records: {offenders}"
