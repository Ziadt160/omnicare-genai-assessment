"""Response normalisation: what shape an answer is allowed to leave in.

Two failures reported from real sessions, both about form rather than truth:
an answer arriving with "### Section 1" and "- **Coverage**:" shown literally,
and the standing risk that a small model answers in JSON while a tool schema
sits in its context.
"""

from __future__ import annotations

import pytest

from libs.guardrails.response import (
    normalize_response,
    strip_markdown,
    unwrap_json,
)


# ------------------------------------------------------------ json answers

@pytest.mark.parametrize("raw,expected", [
    ('{"response": "Sudden pipe bursts are covered up to $25,000."}',
     "Sudden pipe bursts are covered up to $25,000."),
    ('{"answer": "Your deductible is $500."}', "Your deductible is $500."),
    ('```json\n{"response": "Flood damage is excluded."}\n```',
     "Flood damage is excluded."),
    ('  {"text": "Claim CLM-8821 is Approved."}  ', "Claim CLM-8821 is Approved."),
])
def test_an_answer_returned_as_json_is_unwrapped(raw: str, expected: str) -> None:
    assert unwrap_json(raw) == expected


def test_the_contracts_own_key_wins_when_several_are_present() -> None:
    """`response` is what the gateway's contract calls it, so a model echoing
    the schema back gets read the way the schema means it."""
    assert unwrap_json('{"response": "the answer", "text": "something else"}') == (
        "the answer"
    )


@pytest.mark.parametrize("raw", [
    "Your deductible is $500.",
    "The limit is $25,000 (see Section 1).",
    # An object with no prose in it. Guessing which field is "the answer" would
    # turn a formatting fix into a content edit.
    '{"limit": 25000, "deductible": 500}',
    # A list is not an answer either.
    '["Section 1", "Section 2"]',
    # Malformed JSON is just text that happens to start with a brace.
    '{"response": "unterminated',
    # A brace mid-sentence is prose.
    'Use {policy_number} as the placeholder in the form.',
])
def test_anything_that_is_not_a_json_answer_is_left_alone(raw: str) -> None:
    assert unwrap_json(raw) == raw


def test_prose_fenced_as_code_is_unfenced() -> None:
    """A stray ``` around an ordinary paragraph turns the answer into a code
    block. Only a fence wrapping the whole answer is touched."""
    assert unwrap_json("```\nYour deductible is $500.\n```") == (
        "Your deductible is $500."
    )


def test_a_fenced_block_inside_a_longer_answer_survives() -> None:
    """There the fence is deliberate - the model is quoting something - and
    unwrapping it would splice code into prose."""
    raw = "Here is the payload:\n\n```json\n{\"a\": 1}\n```\n\nAnything else?"
    assert unwrap_json(raw) == raw


# --------------------------------------------------------------- markdown

def test_markdown_becomes_speakable_prose() -> None:
    """For voice there is no renderer at all, so a TTS engine would be handed
    "###" and "**" to pronounce."""
    spoken = strip_markdown(
        "### Section 1: Home Water Damage Coverage\n"
        "- **Coverage**: sudden pipe bursts.\n"
        "- Up to $25,000 with a $500 deductible.\n"
    )
    for marker in ("#", "**", "- "):
        assert marker not in spoken, marker
    assert "Section 1: Home Water Damage Coverage." in spoken
    assert "$25,000" in spoken and "$500" in spoken


def test_a_bullet_keeps_its_content() -> None:
    """Only the marker is unspeakable. The content of the list is the answer,
    so dropping the line would drop what was said."""
    assert "sudden pipe bursts" in strip_markdown("- sudden pipe bursts")


def test_a_heading_gains_terminal_punctuation() -> None:
    """Without it TTS runs the heading straight into the line beneath it."""
    assert strip_markdown("## Summary\nYou pay $500.").startswith("Summary.")


def test_a_heading_that_already_ends_in_punctuation_is_not_doubled() -> None:
    assert strip_markdown("## Are you covered?").rstrip() == "Are you covered?"


def test_links_keep_their_text() -> None:
    assert strip_markdown("See [Section 1](http://x/policy).") == "See Section 1."


# ---------------------------------------------------------------- channel

def test_text_keeps_its_markdown() -> None:
    """The chat UI renders headings, lists and emphasis. Flattening them there
    would throw away structure the reader can use."""
    raw = "### Summary\n- Covered up to $25,000."
    assert normalize_response(raw, "text") == raw


def test_voice_keeps_none_of_it() -> None:
    spoken = normalize_response("### Summary\n- Covered up to $25,000.", "voice")
    assert "#" not in spoken and "- " not in spoken
    assert "$25,000" in spoken


def test_both_channels_unwrap_json() -> None:
    """A serialized object is unreadable whether it is displayed or spoken."""
    raw = '{"response": "Your deductible is $500."}'
    for channel in ("text", "voice"):
        assert normalize_response(raw, channel) == "Your deductible is $500."


def test_an_empty_answer_is_left_empty() -> None:
    assert normalize_response("", "text") == ""
