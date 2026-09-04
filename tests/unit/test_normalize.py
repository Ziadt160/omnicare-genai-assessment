"""Spoken-form normalization: the case table.

Written before the implementation. Every row here is a transcript shape STT
actually produces for the two ids in the fixture data.
"""

import pytest

from libs.guardrails.normalize import (
    normalize_claim_id,
    normalize_policy_number,
    phonetic_readback,
    spoken_amounts,
)


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # already canonical
        ("POL-1092", "POL-1092"),
        ("my policy is POL-1092", "POL-1092"),
        ("pol 1092", "POL-1092"),
        # letters spelled out
        ("P O L 1092", "POL-1092"),
        ("P.O.L. 1092", "POL-1092"),
        # numbers spoken in pairs - the common case
        ("policy ten ninety two", "POL-1092"),
        ("pol ten ninety-two", "POL-1092"),
        # digit by digit
        ("policy number one zero nine two", "POL-1092"),
        ("policy one oh nine two", "POL-1092"),
        # no identifier present
        ("what is covered for water damage", None),
        ("I need to file a claim", None),
    ],
)
def test_normalize_policy_number(spoken: str, expected: str | None) -> None:
    assert normalize_policy_number(spoken) == expected


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("CLM-8821", "CLM-8821"),
        ("clm 8821", "CLM-8821"),
        ("claim eighty eight twenty one", "CLM-8821"),
        ("claim eighty-eight twenty-one", "CLM-8821"),
        ("claim number C L M 8 8 2 1", "CLM-8821"),
        ("clm 9014", "CLM-9014"),
        ("claim ninety fourteen", "CLM-9014"),
        ("claim nine oh one four", "CLM-9014"),
        # five digits is not a claim id - must not silently truncate
        ("claim ninety zero fourteen", None),
        ("what is my coverage", None),
    ],
)
def test_normalize_claim_id(spoken: str, expected: str | None) -> None:
    assert normalize_claim_id(spoken) == expected


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("CLM-8821", "C-L-M, eight eight two one"),
        ("POL-1092", "P-O-L, one zero nine two"),
    ],
)
def test_phonetic_readback(identifier: str, expected: str) -> None:
    """A user can only verify by ear what is spoken verifiably."""
    assert phonetic_readback(identifier) == expected


# ------------------------------------------ an identifier ends where it ends

@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("file a claim on policy ten ninety two for twelve hundred dollars",
         "POL-1092"),
        ("policy ten ninety two, the amount is fifteen hundred", "POL-1092"),
        ("my policy is ten ninety two", "POL-1092"),
        ("policy number ten ninety two please", "POL-1092"),
        ("policy ten ninety two", "POL-1092"),
    ],
)
def test_a_spoken_policy_number_survives_a_trailing_amount(spoken, expected) -> None:
    """Everything after the cue was being converted, so the amount ran into the
    identifier: "policy ten ninety two for twelve hundred dollars" produced
    109212, failed the four-digit check and returned None.

    That is the most ordinary sentence a caller says when filing a claim, so
    spoken policy numbers were effectively never normalised - the model was
    left to guess the digits, and the eval only passed because the scripted
    model was handed the right answer.
    """
    assert normalize_policy_number(spoken) == expected


def test_a_claim_id_survives_a_trailing_amount() -> None:
    assert normalize_claim_id(
        "check claim C L M eight eight two one for me"
    ) == "CLM-8821"


def test_a_policy_number_is_not_read_as_a_claim_id() -> None:
    """"file a claim on policy POL 1092" was yielding CLM-1092: the word
    "claim" is a cue, and the digits after it belong to the policy."""
    assert normalize_policy_number("file a claim on policy POL 1092") == "POL-1092"
    assert normalize_claim_id("file a claim on policy POL 1092") is None


# ---------------------------------------------------------- spoken amounts

@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("twelve hundred dollars", 1200),
        ("fifteen hundred", 1500),
        ("one thousand two hundred", 1200),
        ("three thousand", 3000),
        ("twenty five hundred", 2500),
        ("nine hundred and fifty", 950),
        ("it is 1500 usd", 1500),
        ("$1,500.00", 1500),
    ],
)
def test_a_spoken_amount_is_understood(said: str, expected: int) -> None:
    """A caller says "twelve hundred dollars", never "1200.00".

    The confirmation gate checks that the amount on a permanent record came
    from the policyholder, so without this every voice claim would be refused
    for an amount the caller had just said out loud.
    """
    assert expected in spoken_amounts(said)


def test_words_that_are_not_amounts_yield_nothing() -> None:
    assert spoken_amounts("my television was stolen") == set()
    assert spoken_amounts("") == set()


def test_a_section_number_is_not_an_amount() -> None:
    """"Section 1" and "Section 2" are everywhere in this domain."""
    assert spoken_amounts("under section two of my policy") == set()


def test_a_small_written_amount_is_an_amount() -> None:
    """CI caught this: "File a water damage claim on POL-1092 for $99" was
    refused, because one floor was being applied to written figures as well as
    spoken ones. A $99 claim is perfectly ordinary."""
    assert 99 in spoken_amounts("File a water damage claim on POL-1092 for $99")
    assert 50 in spoken_amounts("it was about 50 dollars")


def test_a_small_spoken_number_is_still_not_an_amount() -> None:
    """The floor stays where the ambiguity is: number words."""
    assert spoken_amounts("under section two of my policy") == set()
    assert spoken_amounts("three items were taken") == set()
