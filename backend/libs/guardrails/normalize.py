"""Spoken-form entity normalization for the voice path.

STT does not return ``POL-1092``. It returns "pol ten ninety two", "P O L
1092", "policy ten ninety-two", "claim eighty-eight twenty-one". Every one of
those must reach the tool layer as a canonical id or Pydantic rejects it and
the agent asks the user to repeat themselves - the single most common way a
voice assistant feels broken.

Pure functions, no I/O. See spec §7; the case table lives in
tests/unit/test_normalize.py.
"""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_LETTERS = {
    "pee": "P", "oh": "O", "el": "L", "ell": "L", "see": "C", "sea": "C",
    "em": "M", "cee": "C",
}

_POLICY_CUES = ("pol", "policy", "policy number", "p o l")
_CLAIM_CUES = ("clm", "claim", "claim number", "c l m")


def _words_to_digits(tokens: list[str]) -> str:
    """Concatenate spoken number groups into a digit string.

    "ten ninety two"        -> "1092"   (10, 92)
    "eighty eight twenty one" -> "8821" (88, 21)
    "one zero nine two"     -> "1092"   (1, 0, 9, 2)
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        # Lower-cased: the letter map may already have turned "oh" into "O",
        # and "oh"/"o" is a spoken zero as often as it is the letter.
        t = tokens[i].lower()
        if t.isdigit():
            out.append(t)
            i += 1
        elif t in _TENS:
            value = _TENS[t]
            nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else None
            if nxt in _UNITS and 1 <= _UNITS[nxt] <= 9:
                value += _UNITS[nxt]
                i += 1
            out.append(str(value))
            i += 1
        elif t in _UNITS:
            out.append(str(_UNITS[t]))
            i += 1
        else:
            i += 1
    return "".join(out)


# Said between the cue and the digits, and carrying none of them.
_FILLER = {"is", "number", "no", "the", "my", "it", "was", "of", "for", "hash"}


def _identifier_run(tokens: list[str]) -> list[str]:
    """The number words that form the identifier, and nothing after them.

    Everything following the cue used to be converted, so the amount ran into
    the identifier: "policy ten ninety two for twelve hundred dollars" became
    109212, failed the four-digit check and yielded nothing. That is the most
    ordinary sentence a caller says when filing a claim.

    An identifier is a contiguous run of number words. A little filler is
    allowed before it starts - "my policy is ten ninety two" - but once the
    digits begin, the first word that is not one of them ends them.
    """
    run: list[str] = []
    for token in tokens:
        word = token.lower().strip(",.;:")
        if word.isdigit() or word in _UNITS or word in _TENS:
            run.append(word)
        elif run:
            break
        elif word not in _FILLER:
            # A non-filler word before any digit means the cue was not
            # introducing an identifier at all.
            break
    return run


def _canonicalize(text: str, prefix: str, cues: tuple[str, ...]) -> str | None:
    lowered = text.lower()

    # Already canonical, or one hyphen/space away from it.
    direct = re.search(rf"\b{prefix}\s*[-\s]?\s*(\d{{4}})\b", lowered)
    if direct:
        return f"{prefix.upper()}-{direct.group(1)}"

    # Letters spelled out: "P O L 1092", "P.O.L. 1092", "see el em 8821"
    spaced = re.sub(r"[.\-]", " ", lowered)
    tokens = spaced.split()
    tokens = [_LETTERS.get(t, t) for t in tokens]
    joined = " ".join(tokens)
    letterwise = re.search(
        rf"\b{' '.join(prefix.upper())}\b(.*)", joined, flags=re.IGNORECASE
    )

    tail: str | None = None
    if letterwise:
        tail = letterwise.group(1)
    else:
        for cue in sorted(cues, key=len, reverse=True):
            idx = joined.find(cue)
            if idx != -1:
                tail = joined[idx + len(cue):]
                break
    if tail is None:
        return None

    digits = _words_to_digits(_identifier_run(tail.split()))
    if len(digits) == 4:
        return f"{prefix.upper()}-{digits}"
    return None


# Multipliers, which is what separates an amount from an identifier: "ten
# ninety two" is a policy number read digit-group by digit-group, while
# "twelve hundred" is a quantity.
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}

# An amount below this is a section number, an item count or a stray digit.
# Matches the floor the graph uses when reading figures out of an answer.
_AMOUNT_FLOOR = 100


def spoken_amounts(text: str) -> set[int]:
    """Every money-sized quantity in a piece of text, spoken or written.

    A caller says "twelve hundred dollars", never "1200.00". The confirmation
    gate refuses an amount the policyholder never gave, so without this every
    voice claim would be turned away for a figure the caller had just said out
    loud.

    Deliberately returns whole dollars and a set, not a single parsed value:
    the question being asked is "did they say this number", not "what did they
    mean", and a sentence can carry several.
    """
    found: set[int] = set()

    for match in re.finditer(r"\d[\d,]*", text):
        digits = match.group().replace(",", "")
        if digits.isdigit() and int(digits) >= _AMOUNT_FLOOR:
            found.add(int(digits))

    total = current = 0
    seen = False
    for raw in re.split(r"[^a-z]+", text.lower()):
        if raw in _UNITS:
            current += _UNITS[raw]
            seen = True
        elif raw in _TENS:
            current += _TENS[raw]
            seen = True
        elif raw in _SCALES:
            scale = _SCALES[raw]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            seen = True
        elif raw == "and" and seen:
            continue
        else:
            if seen and total + current >= _AMOUNT_FLOOR:
                found.add(total + current)
            total = current = 0
            seen = False
    if seen and total + current >= _AMOUNT_FLOOR:
        found.add(total + current)

    return found


def normalize_policy_number(text: str) -> str | None:
    """Extract a canonical ``POL-####`` from spoken or typed text, or None."""
    return _canonicalize(text, "pol", _POLICY_CUES)


def normalize_claim_id(text: str) -> str | None:
    """Extract a canonical ``CLM-####`` from spoken or typed text, or None."""
    return _canonicalize(text, "clm", _CLAIM_CUES)


def phonetic_readback(identifier: str) -> str:
    """Render an id so a human can verify it by ear.

    ``CLM-8821`` -> ``"C-L-M, eight eight two one"``. A user can only confirm
    what is spoken verifiably; "clm eighty-eight twenty-one" is exactly as
    ambiguous as the transcript that produced it.
    """
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    prefix, _, digits = identifier.partition("-")
    spelled = "-".join(prefix.upper())
    spoken = " ".join(digit_words.get(d, d) for d in digits)
    return f"{spelled}, {spoken}"
