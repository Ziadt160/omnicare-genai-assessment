"""Contracts for the claims domain.

The supplied ``mock_claims.json`` stores ``"amount": 3500.00`` as a JSON
*number*. ``Decimal`` gives correct money validation, but Pydantic serializes
it to a string by default - which would silently change the shape of the file
the assessment tells us to append to. The field serializer below keeps it a
number. ``tests/unit/test_claims_roundtrip.py`` asserts this.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

ClaimType = Literal[
    "Water Damage",
    "Personal Property",
    "Liability",
    "Auto",
    "Medical",
]
ClaimStatus = Literal["Submitted", "Under Review", "Approved", "Denied"]

POLICY_NUMBER_PATTERN = r"^POL-\d{4}$"
CLAIM_ID_PATTERN = r"^CLM-\d{4}$"


class GetClaimStatusArgs(BaseModel):
    """Tool arguments for ``get_claim_status``. This *is* the LLM's JSON schema."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(
        pattern=CLAIM_ID_PATTERN,
        description=(
            "Claim identifier the policyholder gave you: the letters CLM, a "
            "hyphen, then four digits. Never supply one they did not say - "
            "asking is correct, guessing looks up someone else's claim."
        ),
    )


class SubmitClaimArgs(BaseModel):
    """Tool arguments for ``submit_claim``. This *is* the LLM's JSON schema.

    Field descriptions reach the model verbatim - they are prompt text, not
    documentation, and changing them changes tool-argument accuracy.
    """

    model_config = ConfigDict(extra="forbid")

    # No `examples` on the fields that carry the policyholder's own data.
    #
    # A tool schema is prompt text, and a model copies what it is shown. Asked
    # to file a theft claim, qwen2.5 announced "Policy Number: POL-1092"
    # without the policyholder ever saying it; POL-1092 appears nowhere in the
    # system prompt - it was this field's example. CLM-8821 and 1200.00 were
    # here too, and all three are live rows in mock_claims.json, so a copied
    # example does not merely invent a value: it files against a real
    # policyholder's policy. The pattern and the description carry the format,
    # and neither can be pasted into a tool call as a value.
    policy_number: str = Field(
        pattern=POLICY_NUMBER_PATTERN,
        description=(
            "The policyholder's own policy number: the letters POL, a hyphen, "
            "then four digits. Take it from what they told you. If they have "
            "not given one, ask - never supply a number yourself."
        ),
    )
    claim_type: ClaimType = Field(
        description="Must be one of the listed categories. Map the user's "
                    "wording onto the closest option and say which you chose.",
    )
    amount: Decimal = Field(
        gt=Decimal("0"),
        le=Decimal("1000000"),
        max_digits=12,
        decimal_places=2,
        description=(
            "The amount the policyholder stated, in USD. Never estimate it, "
            "never round it, and never fill in a placeholder: if they have not "
            "given a figure, ask for one. This becomes a permanent record."
        ),
    )
    description: str = Field(
        min_length=10,
        max_length=1000,
        description="What happened, in the policyholder's own words.",
    )

    @field_serializer("amount")
    def _amount_as_number(self, v: Decimal) -> float:
        return float(v)


class Claim(BaseModel):
    """A claim record as stored in ``mock_claims.json``."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=CLAIM_ID_PATTERN)
    policy_number: str = Field(pattern=POLICY_NUMBER_PATTERN)
    claim_type: ClaimType
    status: ClaimStatus
    amount: Decimal = Field(gt=Decimal("0"), max_digits=12, decimal_places=2)

    # Present on every record, empty on the two seeded ones. The brief's
    # sample data omitted it, but submit_claim collects a description and
    # discarding it would make the stored claim less useful than the request
    # that created it. Empty rather than null keeps one shape for all rows.
    description: str = Field(default="", max_length=1000)

    @field_serializer("amount")
    def _amount_as_number(self, v: Decimal) -> float:
        return float(v)
