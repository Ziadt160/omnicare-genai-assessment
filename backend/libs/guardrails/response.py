"""What shape an answer is allowed to leave in.

Three things a model does to the *form* of an answer, none of which the prompt
can reliably stop, and all of which are checkable without asking it:

  * **It answers in JSON.** Asked a question while a tool schema sits in its
    context, a small model sometimes replies ``{"response": "..."}`` or wraps
    prose in a ```json fence. The gateway hands that straight through and the
    policyholder reads a serialized object.
  * **It writes markdown nobody renders.** The chat UI renders a deliberately
    tiny subset, and the voice channel renders none at all - so "### Section 1"
    arrives on screen with the hashes showing, and a TTS engine is handed
    asterisks to pronounce.
  * **It fences prose it never meant as code.** A stray ``` around an ordinary
    paragraph turns an answer into a code block.

Normalising rather than rejecting, for the same reason `ground` rewrites
instead of raising: a formatting quirk is not worth turning into a 500, and an
answer the policyholder cannot read is a failure whether or not it was the
model's fault.

The channel decides how much survives. Text keeps markdown, because the chat UI
renders it. Voice keeps none, because a screen reader is not what is on the
other end - somebody is listening.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Keys a model uses when it wraps an answer in an object. Ordered: the first
# one present wins, so `{"response": "...", "text": "..."}` reads the field the
# gateway's own contract uses.
_TEXT_KEYS = ("response", "answer", "text", "content", "message", "output", "result")

_FENCE_RE = re.compile(
    r"^\s*```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)\r?\n?\s*```\s*$",
    re.S,
)


def _unfence(text: str) -> tuple[str, str]:
    """Split a whole-text code fence into ``(language, body)``.

    Only a fence wrapping the *entire* answer is touched. A fenced block inside
    a longer reply is deliberate - the model is quoting something - and
    unwrapping it would splice code into prose.
    """
    match = _FENCE_RE.match(text)
    return (match.group(1).lower(), match.group(2)) if match else ("", text)


def unwrap_json(text: str) -> str:
    """The prose inside a JSON answer, or the text unchanged.

    Conservative on purpose. It unwraps only when the whole answer parses as an
    object carrying a recognised text field: anything else - a JSON array, an
    object of data with no prose in it, a sentence that merely contains a brace
    - is left exactly as the model wrote it. Guessing which field of an
    unfamiliar object is "the answer" would turn a formatting fix into a
    content edit.
    """
    language, body = _unfence(text)
    candidate = body.strip()
    if not candidate.startswith("{"):
        # A fenced non-JSON block is still worth unfencing when it wraps the
        # whole answer - prose in a ``` block renders as code.
        return body.strip() if language in ("", "text", "markdown", "md") else text

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return text

    if not isinstance(parsed, dict):
        return text

    for key in _TEXT_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return text


# --------------------------------------------------------------- markdown

_MD_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\r?\n?")
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*(.+?)[ \t]*#*[ \t]*$", re.M)
_MD_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.M)
_MD_ORDERED = re.compile(r"^[ \t]*(\d+)[.)][ \t]+", re.M)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_MD_ITALIC = re.compile(r"(?<![*\w])[*_]([^*_\n]+)[*_](?![*\w])")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE = re.compile(r"`([^`\n]+)`")
_MD_RULE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)


def strip_markdown(text: str) -> str:
    """Markdown reduced to speakable prose.

    For voice, where there is no renderer at all. The prompt already asks for
    short spoken answers and the model writes markdown anyway - it is what
    models do - so a heading becomes a sentence and emphasis simply goes away,
    rather than a TTS engine being handed "###" and "**" to pronounce.

    A bullet becomes a sentence rather than being deleted with its line: the
    content of the list is the answer, and only the marker is unspeakable.
    """
    text = _MD_FENCE.sub("", text)
    text = _MD_RULE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    # A heading is a sentence when spoken, so it gets terminal punctuation -
    # without it TTS runs the heading straight into the line beneath it.
    text = _MD_HEADING.sub(lambda m: _as_sentence(m.group(1)), text)
    text = _MD_BULLET.sub("", text)
    text = _MD_ORDERED.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _as_sentence(heading: str) -> str:
    heading = heading.strip()
    return heading if heading.endswith((".", "!", "?", ":")) else f"{heading}."


# ----------------------------------------------------- structured answers

@dataclass(frozen=True)
class Answer:
    """A model reply, unpacked.

    `citations` and `confidence` are what the model *claimed*, and neither is
    believed on its own. The citation is intersected with what retrieval
    returned, and the confidence is capped by whether there was any evidence
    for the answer at all - see `ground`. Both raw values are kept so the two
    can be compared: a model reporting 0.95 on an answer the system had to cap
    to 0.3 is the interesting case, and averaging it away would hide it.
    """

    text: str
    confidence: float | None = None
    citations: tuple[str, ...] = ()
    unknown: bool = False
    structured: bool = False
    # The reply was a serialized tool call rather than an answer. Recorded so
    # the caller can substitute something a person can read, and so the rate it
    # happens at is measurable rather than guessed at.
    tool_call_as_text: bool = False


_ANSWER_KEYS = ("answer", "response", "text", "content", "message")


def _as_confidence(value: object) -> float | None:
    """A number in [0, 1], or None.

    Percentages are accepted because models write them: "confidence": 85 means
    0.85 to everything except a comparison against 1.0. Anything unparseable
    becomes None rather than a default, since a made-up confidence is exactly
    what this field must not carry.
    """
    if isinstance(value, bool) or value is None:
        return None

    percent = False
    if isinstance(value, str):
        value = value.strip()
        percent = value.endswith("%")
        value = value.rstrip("%").strip()
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    # Only unmistakable percentages are rescaled. Dividing everything above 1
    # turned a claimed 1.4 - a model overshooting a 0-1 scale - into 0.014, an
    # emphatic "no confidence at all" from a model that meant the opposite.
    # Between 1 and 10 there is no way to tell 5% from a 1-5 rating, so those
    # clamp to 1.0 rather than being guessed at.
    if percent or number >= 10:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def parse_answer(text: str) -> Answer:
    """Unpack a structured reply, or fall back to treating it as prose.

    The fallback is not a degraded mode, it is the common one: a 7B asked for
    JSON produces it most of the time and prose the rest, and an assistant that
    answered correctly in prose must not be discarded for formatting. So a
    reply that does not parse is the answer, with no confidence claimed -
    `structured` records which happened, so the gap between them is measurable
    rather than assumed.
    """
    if not text or not text.strip():
        return Answer(text=text or "")

    _language, body = _unfence(text)
    candidate = body.strip()
    if not candidate.startswith("{"):
        return Answer(text=text.strip())

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return Answer(text=text.strip())
    if not isinstance(parsed, dict):
        return Answer(text=text.strip())

    if "name" in parsed and "arguments" in parsed and not (
        set(_ANSWER_KEYS) & set(parsed)
    ):
        # A tool call written out as text instead of being called.
        #
        # Observed the moment the envelope was introduced: told to answer in
        # JSON, qwen2.5 began emitting {"name": "search_policy_documents",
        # "arguments": {...}} as its reply. Mixing a JSON output format with
        # function calling does this to small models - both are "produce a JSON
        # object" and it stops distinguishing them.
        #
        # Reported as unknown rather than shown. The machinery is not an
        # answer, and passing it through puts a serialized tool call in front
        # of somebody asking whether their kitchen is covered.
        return Answer(
            text="",
            confidence=0.0,
            unknown=True,
            structured=True,
            tool_call_as_text=True,
        )

    answer = next(
        (parsed[k] for k in _ANSWER_KEYS if isinstance(parsed.get(k), str)), None
    )
    if answer is None:
        # An object with no prose in it. Guessing which field is the answer
        # would turn a formatting fix into a content edit.
        return Answer(text=text.strip())

    raw_citations = parsed.get("citations") or parsed.get("citation") or []
    if isinstance(raw_citations, str):
        raw_citations = [raw_citations]
    citations = tuple(
        c.strip() for c in raw_citations if isinstance(c, str) and c.strip()
    )

    unknown = parsed.get("unknown")
    return Answer(
        text=answer.strip(),
        confidence=_as_confidence(parsed.get("confidence")),
        citations=citations,
        unknown=bool(unknown) if unknown is not None else False,
        structured=True,
    )


def normalize_response(text: str, channel: str = "text") -> str:
    """The answer, in the form the channel can actually present.

    Both channels get the JSON unwrapping, because a serialized object is
    unreadable whether it is displayed or spoken. Only voice gets the markdown
    stripped: the chat UI renders headings, lists and emphasis, and flattening
    them there would throw away structure the reader can use.
    """
    if not text:
        return text
    normalized = unwrap_json(text)
    if channel == "voice":
        normalized = strip_markdown(normalized)
    return normalized.strip()
