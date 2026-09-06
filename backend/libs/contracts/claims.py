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

# The categories a claim can be filed under: exactly the sections the policy
# document has, and that is the point.
#
# This list is prompt text. It reaches the model verbatim as `submit_claim`'s
# schema, and while it also named "Liability", "Auto" and "Medical", a
# policyholder asking what their policy covered was told it included "Fire
# Damage, Personal Property, Liability, Auto, Medical". Four of those five came
# straight from here. The model was not inventing from nowhere - it was reading
# a list of filing categories as a list of cover, which is a fair reading of a
# list it was shown and never told the purpose of.
#
# Narrowing it to what the policy actually pays for removes the reading rather
# than arguing with it. A loss the policy does not cover is then not a claim
# type to be mapped onto the nearest option - it is something to tell the
# policyholder plainly, which `search_policy_documents` can do and this cannot.
#
# The cost is that this couples the contract to one policy document. Deriving
# the list from the document's own sections is the right fix and a larger one;
# until then, two names that match the policy are more honest than five that
# mostly do not.
ClaimType = Literal[
    "Water Damage",
    "Personal Property",
]
ClaimStatus = Literal["Submitted", "Under Review", "Approved", "Denied"]

POLICY_NUMBER_PATTERN = r"^POL-\d{4}$"
CLAIM_ID_PATTERN = r"^CLM-\d{4}$"


class GetClaimStatusArgs(BaseModel):
    """Tool arguments for ``get_claim_status``. This *is* the LLM's JSON schema."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str = Field(
        pattern=CLAIM_ID_PATTERN,
        description=(
            "Claim identifier the policyholder gave you: the letters CLM, a "
            "hyphen, then four digits. Never supply one they did not say - "
            "asking is correct, guessing looks up someone else's claim."
        ),
    )


class EstimateClaimPaymentArgs(BaseModel):
    """Tool arguments for ``estimate_claim_payment``. This *is* the LLM's JSON
    schema.

    Deliberately two fields and no policy number. An estimate reads the policy
    document, which is the same for every policyholder, so asking for a policy
    number before answering "what would I pay on a $35,000 burst pipe?" wastes
    the turn - the same reasoning as rule 5 in the system prompt.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_type: ClaimType = Field(
        description=(
            "How to file this claim. These are filing categories, not a "
            "statement of cover - only search_policy_documents can say what "
            "the policy covers. A loss that is neither of these is not a "
            "claim to be filed under the nearer one: tell the policyholder "
            "the policy does not cover it."
        ),
    )
    amount: Decimal = Field(
        gt=Decimal("0"),
        le=Decimal("1000000"),
        max_digits=12,
        decimal_places=2,
        description=(
            "The loss or repair amount the policyholder stated, in USD. Never "
            "estimate it and never round it: if they have not given a figure, "
            "ask for one. A split computed from a number nobody said is a "
            "number nobody can act on."
        ),
    )


class SubmitClaimArgs(BaseModel):
    """Tool arguments for ``submit_claim``. This *is* the LLM's JSON schema.

    Field descriptions reach the model verbatim - they are prompt text, not
    documentation, and changing them changes tool-argument accuracy.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

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
        description=(
            "How to file this claim. These are filing categories, not a "
            "statement of cover - only search_policy_documents can say what "
            "the policy covers. A loss that is neither of these is not a "
            "claim to be filed under the nearer one: tell the policyholder "
            "the policy does not cover it."
        ),
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

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

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
