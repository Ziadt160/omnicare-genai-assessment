"""Structured answers, abstention, and what a confidence number is worth.

A model's estimate of its own reliability is the least trustworthy number it
produces: fabricating and recalling feel identical from the inside, so it
reports 0.9 on both. Published unqualified next to an invented answer, that
number does not inform the reader - it endorses the invention.

So the claim is treated as a ceiling, lowered only by things the system can
check for itself, and the raw claim is kept alongside so the two can be
compared. These cases pin that asymmetry: nothing may raise a confidence, and
an absent one is never invented.
"""

from __future__ import annotations

import pytest

from agent.app.graph.nodes import _confidence_for
from libs.guardrails.response import parse_answer


# ------------------------------------------------------------ the envelope

def test_a_structured_reply_is_unpacked() -> None:
    a = parse_answer(
        '{"answer": "Covered up to $25,000.", "confidence": 0.9, '
        '"citations": ["sample_policy.md \\u00a7 Section 1"], "unknown": false}'
    )
    assert a.text == "Covered up to $25,000."
    assert a.confidence == 0.9
    assert a.citations == ("sample_policy.md § Section 1",)
    assert a.unknown is False
    assert a.structured is True


def test_prose_is_still_an_answer() -> None:
    """The fallback is the common path, not a degraded one: a 7B asked for JSON
    produces it most of the time and prose the rest, and an assistant that
    answered correctly in prose must not be discarded over formatting."""
    a = parse_answer("Your deductible is $500.")
    assert a.text == "Your deductible is $500."
    assert a.confidence is None
    assert a.structured is False


def test_a_fenced_envelope_is_unpacked() -> None:
    a = parse_answer('```json\n{"answer": "Excluded.", "confidence": 0.4}\n```')
    assert a.text == "Excluded." and a.confidence == 0.4


@pytest.mark.parametrize("raw,expected", [
    ('{"answer": "x", "confidence": 85}', 0.85),      # a percentage
    ('{"answer": "x", "confidence": "90%"}', 0.9),
    ('{"answer": "x", "confidence": "0.7"}', 0.7),    # a string
    # Overshooting a 0-1 scale is clamped, not rescaled: dividing this by 100
    # would turn an emphatic answer into 0.014.
    ('{"answer": "x", "confidence": 1.4}', 1.0),
    ('{"answer": "x", "confidence": -2}', 0.0),
    ('{"answer": "x", "confidence": "high"}', None),  # unparseable, not a guess
    ('{"answer": "x"}', None),
])
def test_confidence_is_read_forgivingly_but_never_invented(raw, expected) -> None:
    assert parse_answer(raw).confidence == expected


def test_an_object_with_no_prose_is_left_alone() -> None:
    """Guessing which field of an unfamiliar object is the answer would turn a
    formatting fix into a content edit."""
    raw = '{"limit": 25000, "deductible": 500}'
    assert parse_answer(raw).text == raw


# ----------------------------------------------------------- the ceiling

def test_a_claim_with_nothing_retrieved_behind_it_is_capped() -> None:
    """The case the whole mechanism exists for. An answer stating policy terms
    with no retrieval this turn is the shape every hallucination in this system
    has taken."""
    value, reason = _confidence_for(
        0.95, unknown=False, retrieved=False, rewritten=False,
        makes_a_policy_claim=True,
    )
    assert value == 0.3
    assert reason and "nothing was retrieved" in reason


def test_a_grounded_claim_gets_the_grounded_value() -> None:
    """Evidence sets it, and a bullish model cannot talk it up: 0.9 claimed on
    a grounded answer still reports the grounded 0.85."""
    value, reason = _confidence_for(
        0.9, unknown=False, retrieved=True, rewritten=False,
        makes_a_policy_claim=True,
    )
    assert value == 0.85 and reason is None


def test_a_rewritten_answer_is_capped() -> None:
    """`ground` removed part of it as unsupported, so whatever the model
    thought it was answering is not what the reader got."""
    value, reason = _confidence_for(
        1.0, unknown=False, retrieved=True, rewritten=True,
        makes_a_policy_claim=True,
    )
    assert value == 0.5 and reason and "removed" in reason


def test_saying_it_does_not_know_reads_as_zero() -> None:
    value, reason = _confidence_for(
        0.8, unknown=True, retrieved=False, rewritten=False,
        makes_a_policy_claim=False,
    )
    assert value == 0.0 and reason and "does not answer this" in reason


def test_a_cap_can_only_lower() -> None:
    """A model that was already unsure must not be talked up by the system
    agreeing with it."""
    value, _ = _confidence_for(
        0.1, unknown=False, retrieved=True, rewritten=False,
        makes_a_policy_claim=True,
    )
    assert value == 0.1


def test_evidence_alone_is_enough_to_score() -> None:
    """The model does not have to say anything. Asking a 7B for a confidence
    number cost accuracy - it started answering jewellery questions with the
    water-damage section - so the number is derived from what the system can
    check, and works the same whether the model volunteers one or not."""
    value, reason = _confidence_for(
        None, unknown=False, retrieved=False, rewritten=False,
        makes_a_policy_claim=True,
    )
    assert value == 0.3
    assert reason and "nothing was retrieved" in reason


def test_a_grounded_answer_scores_without_the_model_saying_anything() -> None:
    value, reason = _confidence_for(
        None, unknown=False, retrieved=True, rewritten=False,
        makes_a_policy_claim=True,
    )
    assert value == 0.85 and reason is None


def test_a_humble_model_is_believed() -> None:
    """Downward only. A model that says 0.2 on a grounded answer knows
    something the evidence check does not, and overriding it upward would
    discard the one direction its self-report is worth anything in."""
    value, _ = _confidence_for(
        0.2, unknown=False, retrieved=True, rewritten=False,
        makes_a_policy_claim=True,
    )
    assert value == 0.2


def test_a_reply_that_claims_nothing_is_not_scored() -> None:
    """"Which policy number is it?" asserts nothing about cover, so there is
    nothing to be confident about. None rather than a number, because a number
    would imply a judgement nobody made."""
    value, reason = _confidence_for(
        None, unknown=False, retrieved=False, rewritten=False,
        makes_a_policy_claim=False,
    )
    assert value is None and reason is None


# ------------------------------------------- what counts as a policy claim

@pytest.mark.parametrize("reply", [
    # Reported from a real session: this came back marked "Low confidence -
    # check this". It asserts nothing about the policy; it offers to talk about
    # one. Labelling a greeting unreliable teaches the reader to ignore the
    # label, which costs more than the label was ever worth.
    "Hello! How can I assist you today - perhaps with a question about your "
    "coverage, checking a claim's status, or filing a new claim?",
    "I'm only able to help with questions about your policy coverage, checking "
    "the status of an existing claim, or filing a new claim.",
    "Sure, I can help you file a claim. I'll need your policy number.",
    "I still need the estimated amount you're claiming. Could you provide that?",
    "Your claim has been submitted. Confirmation ID: CLM-9027.",
    "Which policy number is it?",
])
def test_a_reply_that_only_mentions_the_topic_is_not_a_policy_claim(reply) -> None:
    from agent.app.graph.nodes import _POLICY_CLAIM_RE

    assert not _POLICY_CLAIM_RE.search(reply)


@pytest.mark.parametrize("reply", [
    "A burst pipe is covered up to $25,000 with a $500 deductible.",
    "Your policy does not cover earthquake damage.",
    "Gradual leaks or flood damage are strictly excluded.",
    "Your deductible is $500.",
    "Jewelry is covered up to $10,000 in total.",
    "That is not covered under this policy.",
])
def test_an_assertion_about_cover_is_a_policy_claim(reply) -> None:
    """These need evidence behind them, and score low without it."""
    from agent.app.graph.nodes import _POLICY_CLAIM_RE

    assert _POLICY_CLAIM_RE.search(reply)
