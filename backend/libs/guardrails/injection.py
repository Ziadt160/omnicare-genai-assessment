"""Layer 1 of the guardrail stack: deterministic prompt-injection screening.

Runs in the graph's ``guard`` node, before any LLM call, so a blocked input
costs zero tokens and zero quota. There is deliberately no classifier model
here - patterns plus the structural rules in layers 2-5 (spec §7) are faster,
free, and testable, and layer 4 (write confirmation) is what makes the worst
case non-catastrophic rather than merely unlikely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

Severity = Literal["block", "flag"]

MAX_INPUT_CHARS = 4000
REFUSAL = (
    "I can't help with that request. I can answer questions about your OmniCare "
    "policy coverage, look up an existing claim, or help you file a new one."
)


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity
    description: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass
class GuardVerdict:
    """Outcome of screening one user message."""

    allowed: bool
    matched: list[str] = field(default_factory=list)
    severity: Severity | None = None
    reason: str | None = None

    @property
    def forces_confirmation(self) -> bool:
        """A flagged input may proceed, but never writes without explicit consent."""
        return self.severity == "flag"


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> tuple[Rule, ...]:
    src = Path(path) if path else Path(__file__).with_name("patterns.yaml")
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    return tuple(
        Rule(
            id=r["id"],
            severity=r["severity"],
            description=r["description"],
            patterns=tuple(re.compile(p) for p in r["patterns"]),
        )
        for r in raw["rules"]
    )


def screen(message: str, *, rules: tuple[Rule, ...] | None = None) -> GuardVerdict:
    """Screen one user message.

    Args:
        message: Raw user text - never trusted, never interpolated into a system
            prompt regardless of this verdict (that is layer 2).

    Returns:
        A verdict. ``allowed=False`` means the graph short-circuits to the
        ``ground`` node with a fixed refusal, so the response shape stays
        uniform whether the turn was answered or blocked.
    """
    if len(message) > MAX_INPUT_CHARS:
        return GuardVerdict(
            allowed=False, matched=["LEN-01"], severity="block",
            reason=f"Input exceeds {MAX_INPUT_CHARS} characters",
        )

    active = rules if rules is not None else load_rules()
    flagged: list[str] = []

    for rule in active:
        for pattern in rule.patterns:
            if pattern.search(message):
                if rule.severity == "block":
                    return GuardVerdict(
                        allowed=False, matched=[rule.id],
                        severity="block", reason=rule.description,
                    )
                flagged.append(rule.id)
                break

    if flagged:
        return GuardVerdict(allowed=True, matched=flagged, severity="flag")
    return GuardVerdict(allowed=True)
