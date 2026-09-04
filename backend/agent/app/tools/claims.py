"""The two operational backend tools.

These docstrings and the Field descriptions in ``SubmitClaimArgs`` are not
documentation - they are the tool schema handed to the model, verbatim, and
they are the highest-leverage text in the codebase. Tool-selection accuracy and
argument extraction in evals/ move when this wording moves.

The pattern each one follows: what it does, when to use it, when NOT to use it
(naming the tool that should be used instead), the exact format of every
argument, and what to do on error. A model that has been told "do not guess"
guesses noticeably less often.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import StructuredTool

from libs.contracts import Claim, GetClaimStatusArgs, SubmitClaimArgs
from libs.errors import ClaimNotFound
from libs.guardrails.normalize import phonetic_readback
from libs.ports.claims import ClaimsRepository
from libs.resilience.policy import idempotency_key


def _closest_ids(wanted: str, known: list[str], limit: int = 3) -> list[str]:
    """Rank known ids by shared trailing digits.

    STT mangles the digits, not the prefix, so proximity on the numeric tail is
    the useful signal. This is what turns a dead-end "not found" into "did you
    mean CLM-8821?" - see docs/adr/0007.
    """
    target = wanted.split("-")[-1]

    def overlap(candidate: str) -> int:
        digits = candidate.split("-")[-1]
        # Not strict: a malformed id of a different length should score low,
        # not raise, so recovery still offers the real ids.
        return sum(1 for a, b in zip(target, digits, strict=False) if a == b)

    return sorted(known, key=lambda c: (-overlap(c), c))[:limit]


def build_claims_tools(
    repo: ClaimsRepository,
    idempotency: Any | None = None,
    *,
    user_id: str = "anonymous",
    turn: int = 0,
) -> list[StructuredTool]:
    """Bind the claims tools to a repository instance.

    A factory rather than module-level decorators, because the repository is
    chosen by Settings (JSON, Postgres or InMemory) and tests need to inject
    the in-memory one without patching.

    Args:
        idempotency: Optional store. When present, a repeated submit_claim with
            identical arguments on the same turn returns the original
            confirmation ID instead of filing a second claim. Without it a
            retried write - a timeout after the server already committed -
            files a duplicate insurance claim, which is the failure that
            actually matters in this domain.
    """

    async def get_claim_status(claim_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            claim: Claim = await repo.get(claim_id)
        except ClaimNotFound as exc:
            suggestions = _closest_ids(claim_id, exc.known_ids)
            return {
                "found": False,
                "claim_id": claim_id,
                "error": f"No claim with id {claim_id}.",
                "did_you_mean": suggestions,
                "readback": [phonetic_readback(s) for s in suggestions],
            }
        return {
            "found": True,
            "claim_id": claim.claim_id,
            "policy_number": claim.policy_number,
            "claim_type": claim.claim_type,
            "status": claim.status,
            "amount": float(claim.amount),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    get_claim_status.__doc__ = """Look up the current status of an insurance claim that has already been filed.

    Use this when the policyholder asks about an existing claim - "what's the
    status of my claim", "has CLM-8821 been approved yet", "did my water damage
    claim go through".

    Do NOT use this to create a claim; use submit_claim for that. Do NOT use
    this to answer coverage questions; use search_policy_documents.

    Args:
        claim_id: Claim identifier in the exact form CLM-#### (for example
            "CLM-8821"). Spoken forms such as "claim eighty-eight twenty-one"
            are normalized before they reach you. If you do not have a claim
            ID, ask the policyholder for it - never guess one.

    Returns:
        When found: policy_number, claim_type, status and amount.
        When not found: found=false plus did_you_mean, a list of the closest
        real claim IDs. Offer those to the policyholder. Never invent a status
        for a claim that was not found.
    """

    async def submit_claim(
        policy_number: str, claim_type: str, amount: Any, description: str
    ) -> dict[str, Any]:
        args = SubmitClaimArgs(
            policy_number=policy_number,
            claim_type=claim_type,  # type: ignore[arg-type]
            amount=amount,
            description=description,
        )

        key = idempotency_key(user_id, "submit_claim", args.model_dump(mode="json"), turn)
        if idempotency is not None:
            existing = await idempotency.get(key)
            if existing is not None:
                # A replay, not a second claim. Return what the first attempt
                # produced rather than filing again.
                claim = await repo.get(existing)
                return {
                    "confirmation_id": claim.claim_id,
                    "status": claim.status,
                    "policy_number": claim.policy_number,
                    "claim_type": claim.claim_type,
                    "amount": float(claim.amount),
                    "readback": phonetic_readback(claim.claim_id),
                    "replayed": True,
                }

        claim = await repo.append(args)
        if idempotency is not None:
            await idempotency.put(key, claim.claim_id)
        return {
            "confirmation_id": claim.claim_id,
            "status": claim.status,
            "policy_number": claim.policy_number,
            "claim_type": claim.claim_type,
            "amount": float(claim.amount),
            "readback": phonetic_readback(claim.claim_id),
        }

    submit_claim.__doc__ = """File a NEW insurance claim on behalf of the policyholder.

    This writes a permanent record. Call it only when the policyholder has
    clearly asked to file a claim AND you have all four arguments. If anything
    is missing, ask for it - do not fill in defaults, do not estimate the
    amount, and do not infer the policy number from earlier context.

    The policyholder is shown a confirmation prompt before this executes. If
    they decline, no record is written and you should say so plainly.

    Args:
        policy_number: Exactly POL-#### (for example "POL-1092").
        claim_type: One of "Water Damage", "Personal Property", "Liability",
            "Auto", "Medical". Map the policyholder's wording onto the closest
            option and tell them which one you chose.
        amount: Claimed amount in USD, greater than 0. Never estimate this - if
            the policyholder has not given a figure, ask for it.
        description: What happened, in the policyholder's own words. At least
            10 characters.

    Returns:
        confirmation_id, the new claim ID, plus a phonetic readback of it. Read
        the confirmation ID back to the policyholder.
    """

    return [
        StructuredTool.from_function(
            coroutine=get_claim_status,
            name="get_claim_status",
            description=get_claim_status.__doc__,
            args_schema=GetClaimStatusArgs,
        ),
        StructuredTool.from_function(
            coroutine=submit_claim,
            name="submit_claim",
            description=submit_claim.__doc__,
            args_schema=SubmitClaimArgs,
        ),
    ]
