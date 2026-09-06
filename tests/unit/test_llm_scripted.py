"""The keyless demo provider's routing.

This is the model a reviewer gets from `docker compose up` with no API key, so
it is the first thing anyone sees. It had no tests, and the cost showed up as a
bug report: a payment question routed nowhere, and the reply came back with no
tool call and no citation - which is indistinguishable from a broken agent.

It is a keyword router, not a model. These tests pin the routing decisions and
nothing about language quality, because there is no language quality to pin.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from libs.adapters.llm_scripted import BANNER, ScriptedProvider


def route(message: str) -> tuple[str, dict]:
    """The tool the router picks for a message, with its arguments."""
    result = ScriptedProvider()._generate([HumanMessage(content=message)])
    calls = result.generations[0].message.tool_calls
    return (calls[0]["name"], calls[0]["args"]) if calls else ("", {})


def answer(payload: dict) -> str:
    """What the router says once a tool has returned."""
    provider = ScriptedProvider()
    messages = [
        HumanMessage(content="anything"),
        AIMessage(content="", tool_calls=[]),
        ToolMessage(content=json.dumps(payload), tool_call_id="c1"),
    ]
    return str(provider._generate(messages).generations[0].message.content)


# ------------------------------------------------------------- payment split

@pytest.mark.parametrize("message", [
    "A pipe burst and the repair is $35,000. How much will you pay and how much will I pay?",
    "How much would OmniCare pay on a $9,000 claim?",
    "My electronics were destroyed - $12,000 worth. What is my share?",
    "The pipe repair was $4,000. How much do I get back?",
    "Who pays what on a $6,000 water damage claim?",
    "A $3,000 leak - what will it cost me?",
])
def test_a_payment_question_reaches_the_estimate_tool(message: str) -> None:
    """The reported failure. None of these share vocabulary with the coverage
    keywords, so before the payment pattern existed they fell through to the
    generic reply - no tool, no citation."""
    name, _args = route(message)
    assert name == "estimate_claim_payment", message


def test_the_subject_of_pay_does_not_matter() -> None:
    """"you", "OmniCare" and "the company" are the same question. A substring
    list grows a row per phrasing and still misses the next one."""
    for who in ("you", "OmniCare", "the company", "my insurer"):
        name, _ = route(f"How much would {who} pay on a $5,000 burst pipe?")
        assert name == "estimate_claim_payment", who


def test_a_payment_question_without_an_amount_is_not_estimated() -> None:
    """A split computed from a figure nobody stated is the one thing this tool
    must never produce, so the question falls back to coverage search."""
    name, _ = route("How much will you pay if a pipe bursts?")
    assert name == "search_policy_documents"


def test_filing_still_wins_over_estimating() -> None:
    """"File a claim ... for $2,000" is a write request that happens to contain
    an amount. Routing it to an estimate would quietly drop the claim."""
    name, _ = route("File a claim on POL-1092 for $2,000 - a burst pipe.")
    assert name == "submit_claim"


def test_a_status_question_still_wins() -> None:
    name, _ = route("What is the status of claim CLM-8821?")
    assert name == "get_claim_status"


@pytest.mark.parametrize("message,expected", [
    ("$4,000 of jewelry stolen. What is my share?", "Personal Property"),
    ("A burst pipe cost $4,000. Who pays what?", "Water Damage"),
])
def test_the_claim_type_is_read_from_the_message(message: str, expected: str) -> None:
    _name, args = route(message)
    assert args["claim_type"] == expected


@pytest.mark.parametrize("message", [
    "How much would you pay on a $9,000 auto claim?",
    "My car was hit - $9,000. What do I pay?",
    "I was injured - $2,000 of medical bills. What do I get back?",
    "Someone sued me for $5,000. Am I covered for liability?",
])
def test_a_loss_the_policy_does_not_cover_goes_to_search(message: str) -> None:
    """`ClaimType` names only what the policy covers, so there is no category
    to file a car under - and mapping it onto the nearest one would answer a
    collision with the burst-pipe figures. Search answers it correctly: the
    policy covers water damage and personal property."""
    name, _args = route(message)
    assert name == "search_policy_documents"


def test_the_company_name_is_not_a_car() -> None:
    """"car" matched inside "OmniCare", which ruled out of scope every question
    that named the company - including the one the payment route exists for."""
    name, _ = route("How much would OmniCare pay on a $9,000 burst pipe?")
    assert name == "estimate_claim_payment"


# ------------------------------------------------------------------ answering

def test_an_estimate_result_is_quoted_not_recomputed() -> None:
    """Extractive on purpose: a wrong number here means the tool is wrong,
    never that the stand-in invented one."""
    text = answer({
        "estimated": True,
        "payment_summary": "Claim amount: $35,000.00\nOmniCare pays: $25,000.00",
        "citation": "sample_policy.md § Section 1: Home Water Damage Coverage",
    })
    assert "OmniCare pays: $25,000.00" in text
    assert "Section 1: Home Water Damage Coverage" in text
    assert text.startswith(BANNER)


def test_an_estimate_with_no_policy_terms_says_so() -> None:
    """This used to fall through to the generic error branch and report "that
    did not work", which reads as a fault rather than as the policy being
    silent on the subject."""
    text = answer({
        "estimated": False, "error": "no_policy_terms", "claim_type": "Auto",
    })
    assert "no coverage limit or deductible" in text
    assert "Auto" in text


def test_an_unroutable_message_still_answers() -> None:
    name, _ = route("hello there")
    assert name == ""
