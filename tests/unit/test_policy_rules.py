"""Reading the numeric rules out of a policy document.

A policy states its own limits in prose - "covered up to $25,000 with a $500
deductible" - and until now nothing read them. A claim for $250,000 against a
section capped at $25,000 was filed without comment, ten times the limit,
because the only thing checking the amount was a model doing arithmetic in its
head.

These are extracted at ingest and travel with the section, the same way the
citation does, so the comparison that matters is done in code against a figure
that came from the document rather than from the model.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libs.policy.rules import (
    RuleKind,
    coverage_limit_for,
    extract_rules,
)

SECTION_1 = (
    "Section 1: Home Water Damage Coverage\n\n"
    "Water damage caused by sudden pipe bursts is covered up to $25,000 with a "
    "$500 deductible. Gradual leaks or flood damage are strictly excluded."
)
SECTION_2 = (
    "Section 2: Personal Property Protection\n\n"
    "Electronics, furniture, and jewelry are covered up to $10,000 total. "
    "Single items exceeding $2,500 require individual appraisal receipts."
)


# ------------------------------------------------------------- extraction

def test_a_coverage_limit_is_read_from_the_prose() -> None:
    rules = extract_rules(SECTION_1)
    limits = [r for r in rules if r.kind == RuleKind.LIMIT]
    assert [r.amount for r in limits] == [Decimal("25000")]


def test_a_deductible_is_read_and_is_not_mistaken_for_the_limit() -> None:
    """Both amounts sit in one sentence. Reading "$500" as the cap would refuse
    every claim over five hundred dollars."""
    rules = extract_rules(SECTION_1)
    by_kind = {r.kind: r.amount for r in rules}

    assert by_kind[RuleKind.LIMIT] == Decimal("25000")
    assert by_kind[RuleKind.DEDUCTIBLE] == Decimal("500")


def test_a_per_item_threshold_is_not_a_coverage_limit() -> None:
    """"Single items exceeding $2,500 require receipts" is a documentation
    rule. Treating it as the cap would refuse a $9,000 claim on a section that
    covers $10,000."""
    rules = extract_rules(SECTION_2)
    by_kind = {r.kind: r.amount for r in rules}

    assert by_kind[RuleKind.LIMIT] == Decimal("10000")
    assert by_kind[RuleKind.PER_ITEM_THRESHOLD] == Decimal("2500")


def test_prose_with_no_figures_yields_no_rules() -> None:
    assert extract_rules("Section 9: Definitions. Terms used in this policy.") == []


def test_every_rule_names_the_wording_it_came_from() -> None:
    """A refusal has to be able to quote the policy back. "Your policy caps
    this at $25,000" is checkable; "the limit is $25,000" is an assertion."""
    limit = next(r for r in extract_rules(SECTION_1) if r.kind == RuleKind.LIMIT)
    assert "25,000" in limit.source_text
    assert "covered up to" in limit.source_text.lower()


# --------------------------------------------------- matching a claim type

def test_a_claim_type_finds_its_section() -> None:
    sections = [("Section 1: Home Water Damage Coverage", SECTION_1),
                ("Section 2: Personal Property Protection", SECTION_2)]

    found = coverage_limit_for("Water Damage", sections)
    assert found is not None
    assert found.amount == Decimal("25000")
    assert found.section_title == "Section 1: Home Water Damage Coverage"

    found = coverage_limit_for("Personal Property", sections)
    assert found is not None
    assert found.amount == Decimal("10000")


def test_a_claim_type_with_no_section_yields_nothing() -> None:
    """Fails open, deliberately. "Liability" appears nowhere in this policy,
    and refusing a claim on a section we guessed at would be worse than not
    checking - the confirmation gate still applies either way. A refusal must
    be able to point at the wording it came from.
    """
    sections = [("Section 1: Home Water Damage Coverage", SECTION_1),
                ("Section 2: Personal Property Protection", SECTION_2)]

    assert coverage_limit_for("Liability", sections) is None
    assert coverage_limit_for("Auto", sections) is None


@pytest.mark.parametrize("claim_type", ["water damage", "WATER DAMAGE", "Water  Damage"])
def test_matching_is_forgiving_about_case_and_spacing(claim_type: str) -> None:
    sections = [("Section 1: Home Water Damage Coverage", SECTION_1)]
    assert coverage_limit_for(claim_type, sections) is not None
