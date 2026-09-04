"""Numeric rules read out of a policy document.

A policy states its own limits in prose - "covered up to $25,000 with a $500
deductible" - and nothing was reading them. A claim for $250,000 against a
section capped at $25,000 was filed without comment, ten times the limit,
because the only thing checking the amount was a model doing arithmetic in its
head. It got it wrong in the other direction too, telling a policyholder that
$1,500 "exceeds $2,500".

So the comparison is not asked of the model at all. The figures are extracted
where the document is read, travel with the section the way the citation does,
and the comparison is arithmetic in code. There is no built-in tool for this in
LangGraph - `prebuilt` offers ToolNode, create_react_agent, ValidationNode and
tools_condition, and nothing arithmetic - and a calculator tool would only move
the judgement rather than remove it: the model would still decide when to call
it and what to do with the answer.

Three kinds of figure appear in the same sentence and mean entirely different
things, which is the whole difficulty:

    "covered up to $25,000 with a $500 deductible"
     ^ the cap                   ^ not the cap

    "Single items exceeding $2,500 require individual appraisal receipts"
                            ^ a documentation rule, not the cap

Reading the deductible as the cap refuses every claim over five hundred
dollars; reading the per-item threshold as the cap refuses a $9,000 claim on a
section that covers $10,000. Each kind is matched by the wording that
distinguishes it, and anything unrecognised is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

_AMOUNT = r"\$\s?([\d,]+(?:\.\d{2})?)"


class RuleKind(str, Enum):
    """What a figure in the policy actually governs."""

    LIMIT = "limit"
    DEDUCTIBLE = "deductible"
    PER_ITEM_THRESHOLD = "per_item_threshold"


# Ordered: the first pattern that matches a figure decides what it is. The
# deductible and threshold patterns are checked first because their wording is
# specific, while "up to $X" is the general case.
_PATTERNS: tuple[tuple[RuleKind, re.Pattern[str]], ...] = (
    (RuleKind.DEDUCTIBLE, re.compile(rf"{_AMOUNT}\s+deductible", re.I)),
    (RuleKind.DEDUCTIBLE, re.compile(rf"deductible\s+of\s+{_AMOUNT}", re.I)),
    (
        RuleKind.PER_ITEM_THRESHOLD,
        re.compile(rf"(?:single\s+)?items?\s+(?:exceeding|over|above)\s+{_AMOUNT}", re.I),
    ),
    (RuleKind.LIMIT, re.compile(rf"covered\s+up\s+to\s+{_AMOUNT}", re.I)),
    (RuleKind.LIMIT, re.compile(rf"up\s+to\s+{_AMOUNT}", re.I)),
    (RuleKind.LIMIT, re.compile(rf"(?:capped|limited)\s+(?:at|to)\s+{_AMOUNT}", re.I)),
    (RuleKind.LIMIT, re.compile(rf"maximum\s+(?:of\s+)?{_AMOUNT}", re.I)),
)


@dataclass(frozen=True)
class CoverageRule:
    """One figure from the policy, with the wording it came from.

    `source_text` is kept so a refusal can quote the policy back. "Your policy
    covers this up to $25,000" is checkable by the policyholder; "the limit is
    $25,000" is just another assertion, which is what this whole layer exists
    to avoid.
    """

    kind: RuleKind
    amount: Decimal
    source_text: str
    section_title: str = ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_rules(text: str, section_title: str = "") -> list[CoverageRule]:
    """Every numeric rule stated in a section, one per kind.

    Matched sentence by sentence so a figure is read with the words around it.
    A section states each kind once; if it somehow states one twice, the first
    wins rather than the largest - guessing which of two caps applies is not
    something to do silently.
    """
    found: dict[RuleKind, CoverageRule] = {}

    for sentence in _sentences(text):
        claimed: set[str] = set()
        for kind, pattern in _PATTERNS:
            for match in pattern.finditer(sentence):
                raw = match.group(1)
                if raw in claimed:
                    # Already accounted for by a more specific pattern - the
                    # $500 in "with a $500 deductible" must not also register
                    # as a limit via "up to".
                    continue
                claimed.add(raw)
                if kind not in found:
                    found[kind] = CoverageRule(
                        kind=kind,
                        amount=Decimal(raw.replace(",", "")),
                        source_text=sentence,
                        section_title=section_title,
                    )

    return list(found.values())


def _normalised(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def coverage_limit_for(
    claim_type: str, sections: list[tuple[str, str]]
) -> CoverageRule | None:
    """The coverage cap governing a claim type, or None if the policy has none.

    Matched by the claim type naming the section - "Water Damage" against
    "Section 1: Home Water Damage Coverage" - rather than by similarity. A
    refusal has to be able to point at the wording it came from, and a
    semantically-nearest section is not that: the embedder ranks Personal
    Property above Home Water Damage for "what is my deductible?", and a cap
    applied from the wrong section would refuse a valid claim while quoting a
    figure that does not govern it.

    Returns None when nothing matches, which is deliberate. "Liability" appears
    nowhere in this policy, and refusing a claim against a section we guessed
    at is worse than not checking - the confirmation gate still applies either
    way.

    Args:
        claim_type: The claim's category, as `submit_claim` received it.
        sections: ``(section_title, section_text)`` for the policy.
    """
    wanted = _normalised(claim_type)
    if not wanted:
        return None

    for title, text in sections:
        if wanted in _normalised(title) or wanted in _normalised(text):
            limit = next(
                (r for r in extract_rules(text, title) if r.kind == RuleKind.LIMIT),
                None,
            )
            if limit is not None:
                return limit
    return None
