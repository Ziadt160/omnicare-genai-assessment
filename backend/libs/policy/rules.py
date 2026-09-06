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

The same figures answer the question a policyholder actually asks, which is not
"what is the limit" but "what do I pay". `settle` turns them into that split -
still arithmetic in code, over figures quoted from the document, for exactly
the reason above.
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


def _unpack(section: tuple[str, ...]) -> tuple[str, str, str]:
    """``(title, text)`` or ``(title, text, source_file)``, normalised.

    The source file is optional because it only became necessary once a
    settlement had to carry a citation - the cap check never needed it. Callers
    that supply two elements keep working and simply produce no citation, which
    is the honest outcome: a citation naming a file nobody named would be a
    fabrication, and this module exists to not make those.
    """
    title, text, *rest = section
    return title, text, (rest[0] if rest else "")


def _governing_section(
    claim_type: str, sections: list[tuple[str, str]]
) -> tuple[str, str, str] | None:
    """The section of the policy that governs a claim type, or None.

    Matched by the claim type naming the section - "Water Damage" against
    "Section 1: Home Water Damage Coverage" - rather than by similarity. A
    refusal or a settlement has to be able to point at the wording it came
    from, and a semantically-nearest section is not that: the embedder ranks
    Personal Property above Home Water Damage for "what is my deductible?",
    and a cap applied from the wrong section would refuse a valid claim while
    quoting a figure that does not govern it.

    A section stating a coverage limit wins over one that merely mentions the
    claim type, so this and `coverage_limit_for` can never disagree about which
    section applies. The fallback to the first textual match matters for a
    section that states a deductible and no cap: it still settles, and
    returning None there would silently drop a deductible the policy states.
    """
    wanted = _normalised(claim_type)
    if not wanted:
        return None

    fallback: tuple[str, str, str] | None = None
    for section in sections:
        title, text, source_file = _unpack(section)
        if wanted not in _normalised(title) and wanted not in _normalised(text):
            continue
        if fallback is None:
            fallback = (title, text, source_file)
        if any(r.kind is RuleKind.LIMIT for r in extract_rules(text, title)):
            return title, text, source_file
    return fallback


def coverage_limit_for(
    claim_type: str, sections: list[tuple[str, str]]
) -> CoverageRule | None:
    """The coverage cap governing a claim type, or None if the policy has none.

    Returns None when nothing matches, which is deliberate. "Liability" appears
    nowhere in this policy, and refusing a claim against a section we guessed
    at is worse than not checking - the confirmation gate still applies either
    way.

    Args:
        claim_type: The claim's category, as `submit_claim` received it.
        sections: ``(section_title, section_text)`` for the policy.
    """
    section = _governing_section(claim_type, sections)
    if section is None:
        return None
    title, text, _source_file = section
    return next(
        (r for r in extract_rules(text, title) if r.kind is RuleKind.LIMIT), None
    )


# -------------------------------------------------------------- exclusions

# An exclusion is prose, not a figure, so it is matched by the words that make
# a sentence a denial. Kept deliberately narrow: "excluded", "not covered" and
# "does not cover" are what an exclusion actually says, and widening this to
# "unless" or "subject to" would start quoting ordinary conditions as if they
# were denials.
_EXCLUSION_RE = re.compile(
    r"\b(?:excluded|not\s+covered|does\s+not\s+cover|no\s+coverage)\b", re.I
)


def exclusions_in(text: str) -> list[str]:
    """The sentences in a section that state an exclusion, verbatim.

    Quoted rather than interpreted. Deciding whether *this particular* loss is
    excluded needs the cause of loss, and a claim type does not carry it: a
    "Water Damage" claim may be a burst pipe (covered) or a flood (not), and
    the two are indistinguishable from the tool arguments. So the exclusion
    travels with the settlement as wording the policyholder can check, and the
    arithmetic never silently asserts an eligibility it cannot establish.
    """
    return [s for s in _sentences(text) if _EXCLUSION_RE.search(s)]


# ------------------------------------------------------------- settlement

_CENTS = Decimal("0.01")


@dataclass(frozen=True)
class Settlement:
    """Who pays what on a claim, computed from the policy's own figures.

    Every field is arithmetic in code over figures read out of the document.
    None of it is asked of the model, for the reason this module exists: a 7B
    told a policyholder that $1,500 "exceeds $2,500", and a settlement is that
    same comparison with someone's money attached to the answer.
    """

    claimed: Decimal
    deductible: Decimal
    limit: Decimal | None
    insurer_pays: Decimal
    policyholder_pays: Decimal
    section_title: str = ""
    source_file: str = ""
    policy_says: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        """The section this split was read out of, in the one citation format.

        Empty when the source file is unknown, and empty is right there: a
        citation promises a human can go and read the thing, and one assembled
        from a filename nobody supplied is a fabrication of exactly the kind
        `ground` exists to strip.

        Built through `format_citation` rather than by interpolation, so a
        settlement's citation and a retrieved chunk's citation cannot drift
        into two conventions - `ground` compares them as strings.
        """
        if not self.source_file or not self.section_title:
            return ""
        from libs.contracts.retrieval import format_citation

        return format_citation(self.source_file, self.section_title)

    @property
    def above_limit(self) -> bool:
        """Whether the cap - rather than only the deductible - is why the
        policyholder is carrying part of this."""
        return self.limit is not None and self.claimed - self.deductible > self.limit

    @property
    def applied_deductible(self) -> Decimal:
        """The deductible actually absorbed by this claim.

        Below the deductible the policyholder pays the loss, not the
        deductible: on a $300 claim against a $500 deductible they are out
        $300, and reporting the $500 as their share would overstate it by two
        hundred dollars while still adding up to nothing sensible.
        """
        return min(self.deductible, self.claimed).quantize(_CENTS)

    @property
    def below_deductible(self) -> bool:
        """Whether the claim is small enough that the policy pays nothing."""
        return self.insurer_pays == 0 and self.deductible >= self.claimed > 0

    @property
    def excess(self) -> Decimal:
        """The part above the cap: the policyholder's share beyond the
        deductible. Stated separately because "you pay $10,000" invites exactly
        the question this answers."""
        return (self.policyholder_pays - self.applied_deductible).quantize(_CENTS)

    def as_dict(self) -> dict[str, object]:
        """Flattened for a tool result. Floats, because this crosses a JSON
        boundary into the model's context - `Decimal` is not serializable there
        and the values are already rounded to cents."""
        return {
            "claimed": float(self.claimed),
            "deductible": float(self.deductible),
            "deductible_applied": float(self.applied_deductible),
            "limit": float(self.limit) if self.limit is not None else None,
            "insurer_pays": float(self.insurer_pays),
            "policyholder_pays": float(self.policyholder_pays),
            "policyholder_excess_above_limit": float(self.excess),
            "above_limit": self.above_limit,
            "below_deductible": self.below_deductible,
            "section": self.section_title,
            "citation": self.citation,
            "policy_says": list(self.policy_says),
            "exclusions": list(self.exclusions),
        }

    def summary(self) -> str:
        """One line per party, in the order a policyholder asks for them.

        Both totals are always stated and they always sum to the claim - see
        `settle`. Rule 8 of the system prompt requires two amounts side by side
        for any comparison the policyholder is expected to check; this is that
        rule applied to money rather than to thresholds.
        """
        lines = [
            f"Claim amount: ${self.claimed:,.2f}",
            f"OmniCare pays: ${self.insurer_pays:,.2f}",
            f"You pay: ${self.policyholder_pays:,.2f}",
        ]
        if self.below_deductible:
            # The decomposition below would read "$300.00 ($300.00
            # deductible)", which is arithmetically fine and tells the
            # policyholder nothing about why. The reason is the whole answer
            # here, so it replaces the parts rather than joining them.
            lines[-1] += (
                f" - the claim is below the ${self.deductible:,.2f} deductible, "
                f"so the policy pays nothing on it"
            )
            return "\n".join(lines)

        parts = []
        if self.applied_deductible:
            parts.append(f"${self.applied_deductible:,.2f} deductible")
        if self.excess > 0 and self.limit is not None:
            parts.append(f"${self.excess:,.2f} above the ${self.limit:,.2f} limit")
        if parts:
            lines[-1] += f" ({' + '.join(parts)})"
        return "\n".join(lines)


def settle(
    claimed: Decimal,
    *,
    limit: Decimal | None = None,
    deductible: Decimal | None = None,
    section_title: str = "",
    source_file: str = "",
    policy_says: tuple[str, ...] = (),
    exclusions: tuple[str, ...] = (),
) -> Settlement:
    """Split a claimed amount between the insurer and the policyholder.

    The deductible comes off the loss first, and the limit then caps what the
    insurer pays::

        insurer      = min(claimed - deductible, limit)
        policyholder = claimed - insurer

    That ordering is the general one, and it is what lets "covered up to
    $25,000" be taken at face value: on a $35,000 loss the insurer pays the
    full $25,000 and the policyholder carries $10,000 - the $500 deductible
    plus $9,500 above the cap. Capping first and subtracting the deductible
    afterwards would pay out $24,500 against a policy that says $25,000, which
    is not what the document says.

    Both figures are optional, because a policy need not state both. With no
    limit the insurer pays everything above the deductible; with no deductible
    it pays up to the limit from the first dollar; with neither, this reports
    the insurer paying in full. Nothing is assumed on the policy's behalf - a
    figure the document does not state is absent, never a default someone
    invented.

    The two shares always sum to `claimed` exactly, which is the invariant
    worth having: a breakdown whose parts do not add up is worse than no
    breakdown, because it still looks authoritative. `policyholder_pays` is
    derived by subtraction rather than computed independently, so it cannot
    drift from the total no matter what the figures are.
    """
    claimed = Decimal(claimed).quantize(_CENTS)
    deductible = Decimal(deductible or 0).quantize(_CENTS)

    # A deductible larger than the loss means the policyholder carries all of
    # it - not that the insurer is owed the difference.
    eligible = max(claimed - deductible, Decimal("0"))
    insurer = eligible if limit is None else min(eligible, Decimal(limit))

    return Settlement(
        claimed=claimed,
        deductible=deductible,
        limit=Decimal(limit).quantize(_CENTS) if limit is not None else None,
        insurer_pays=insurer.quantize(_CENTS),
        policyholder_pays=(claimed - insurer).quantize(_CENTS),
        section_title=section_title,
        source_file=source_file,
        policy_says=policy_says,
        exclusions=exclusions,
    )


def settlement_for(
    claim_type: str, amount: Decimal, sections: list[tuple[str, str]]
) -> Settlement | None:
    """The payment split for a claim, read out of the policy, or None.

    None means the policy states nothing that governs this claim type - no
    section names it, or the section that does states no figures. Returning a
    settlement anyway would mean inventing the terms, and "OmniCare pays
    $35,000" asserted from an absence is the worst of the failure modes this
    module exists to prevent. The caller says so plainly instead.

    Args:
        claim_type: The claim's category, as the tools received it.
        amount: The claimed amount in USD.
        sections: ``(section_title, section_text)`` for the policy.
    """
    section = _governing_section(claim_type, sections)
    if section is None:
        return None

    title, text, source_file = section
    rules = {r.kind: r for r in extract_rules(text, title)}
    limit = rules.get(RuleKind.LIMIT)
    deductible = rules.get(RuleKind.DEDUCTIBLE)
    if limit is None and deductible is None:
        return None

    return settle(
        Decimal(amount),
        limit=limit.amount if limit else None,
        deductible=deductible.amount if deductible else None,
        section_title=title,
        source_file=source_file,
        # The wording each figure came from, deduplicated - a limit and a
        # deductible stated in one sentence get quoted once, not twice.
        policy_says=tuple(
            dict.fromkeys(r.source_text for r in (limit, deductible) if r)
        ),
        exclusions=tuple(exclusions_in(text)),
    )
