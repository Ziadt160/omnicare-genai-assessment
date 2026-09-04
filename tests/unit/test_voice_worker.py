"""The voice worker's logic, without LiveKit, a microphone or an SFU.

The worker was written twice: first against the livekit-agents 0.12 API from
memory, then against 1.7 after checking the installed package. These tests
exist so the next version bump fails here rather than in a room with a caller
on the line.

What is exercised: the shim that presents the agent as an `llm.LLM`, transcript
normalization, and the split between what is spoken and what is shown. What is
not: VAD, STT, TTS and the SFU, which are third-party and need audio.
"""

from __future__ import annotations

import pytest

from libs.contracts import RunEvent
from voice.app.agent_llm import TOOL_FILLERS, OmniCareLLM, _AgentStream
from voice.app.worker import browser_message, normalize_transcript

CITATION = "sample_policy.md § Section 1: Home Water Damage Coverage"


class ScriptedQueue:
    """Accepts a job and replays a fixed event sequence, like the agent."""

    def __init__(self, events: list[RunEvent]) -> None:
        self.events = events
        self.jobs: list[dict] = []

    async def enqueue(self, run_id: str, job: dict, deadline_s: float) -> None:
        self.jobs.append({"run_id": run_id, **job})

    async def subscribe(self, run_id: str, timeout_s: float):
        for event in self.events:
            yield event.model_copy(update={"run_id": run_id})


def events(*pairs) -> list[RunEvent]:
    return [
        RunEvent(run_id="", type=t, seq=i, payload=p)  # type: ignore[arg-type]
        for i, (t, p) in enumerate(pairs)
    ]


# ------------------------------------------------------------- the shim

async def test_a_turn_is_enqueued_on_the_voice_channel() -> None:
    """`channel="voice"` is the single field that changes graph behaviour - it
    selects the confirmation tier and the phonetic read-back. If it stops being
    sent, voice silently degrades to text semantics."""
    queue = ScriptedQueue(events(("token", {"text": "Yes."}), ("done", {})))
    shim = OmniCareLLM(queue, user_id="caller-1", conversation_id="room-9")
    shim.last_confidence = 0.42

    stream = _AgentStream(shim, chat_ctx=_ctx("am I covered?"), tools=[],
                          conn_options=_conn())
    await _drain(stream)

    job = queue.jobs[0]
    assert job["channel"] == "voice"
    assert job["user_id"] == "caller-1"
    assert job["conversation_id"] == "room-9"
    assert job["stt_confidence"] == 0.42


async def test_tokens_reach_the_session_as_chunks() -> None:
    queue = ScriptedQueue(events(
        ("token", {"text": "Covered up to $25,000."}), ("done", {}),
    ))
    shim = OmniCareLLM(queue, user_id="c", conversation_id="r")
    stream = _AgentStream(shim, chat_ctx=_ctx("am I covered?"), tools=[],
                          conn_options=_conn())

    assert "Covered up to $25,000." in await _drain(stream)


async def test_a_tool_call_speaks_a_filler_immediately() -> None:
    """Several seconds of silence on a call reads as a dropped connection. The
    filler is the difference between a demo that feels alive and one that feels
    broken."""
    queue = ScriptedQueue(events(
        ("tool_start", {"name": "get_claim_status"}),
        ("token", {"text": "It is Approved."}),
        ("done", {}),
    ))
    shim = OmniCareLLM(queue, user_id="c", conversation_id="r")
    spoken = await _drain(
        _AgentStream(shim, chat_ctx=_ctx("status of CLM-8821?"), tools=[],
                     conn_options=_conn())
    )

    assert TOOL_FILLERS["get_claim_status"] in spoken
    assert spoken.index(TOOL_FILLERS["get_claim_status"]) < spoken.index("It is Approved.")


async def test_citations_are_shown_not_spoken() -> None:
    """Reading "sample_policy.md section 1" aloud is noise; the caller should
    still see it."""
    queue = ScriptedQueue(events(
        ("token", {"text": "Yes, that is covered."}),
        ("sources", {"sources": [CITATION]}),
        ("done", {}),
    ))
    shim = OmniCareLLM(queue, user_id="c", conversation_id="r")
    spoken = await _drain(
        _AgentStream(shim, chat_ctx=_ctx("covered?"), tools=[], conn_options=_conn())
    )

    assert "sample_policy.md" not in spoken
    assert browser_message(
        RunEvent(run_id="r", type="sources", seq=0, payload={"sources": [CITATION]})
    ) == {"type": "answer", "text": "", "sources": [CITATION]}


async def test_a_confirmation_is_spoken() -> None:
    """The read-back is the whole point of the tier-2 gate - if it is not
    spoken, an irreversible write is confirmed by someone who never heard the
    details."""
    readback = "I'm about to file a Water Damage claim on policy P-O-L, one zero nine two."
    queue = ScriptedQueue(events(("confirm", {"readback": readback}), ("done", {})))
    shim = OmniCareLLM(queue, user_id="c", conversation_id="r")

    assert readback in await _drain(
        _AgentStream(shim, chat_ctx=_ctx("file a claim"), tools=[], conn_options=_conn())
    )


async def test_a_silent_run_still_says_something() -> None:
    """A turn that produced nothing sayable is still a turn the caller is
    waiting on. Silence is the one unacceptable outcome on a voice channel."""
    queue = ScriptedQueue(events(("done", {})))
    shim = OmniCareLLM(queue, user_id="c", conversation_id="r")

    assert (await _drain(
        _AgentStream(shim, chat_ctx=_ctx("hello"), tools=[], conn_options=_conn())
    )).strip() != ""


async def test_an_error_is_spoken_not_swallowed() -> None:
    queue = ScriptedQueue(events(("error", {"message": "boom"}), ("done", {})))
    shim = OmniCareLLM(queue, user_id="c", conversation_id="r")
    spoken = (await _drain(
        _AgentStream(shim, chat_ctx=_ctx("hi"), tools=[], conn_options=_conn())
    )).lower()

    assert "something went wrong" in spoken
    assert "boom" not in spoken, "internal detail must not be read to a caller"


# ------------------------------------------------------ transcript handling

@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("what is the status of claim eighty eight twenty one", "CLM-8821"),
        ("file a claim on policy ten ninety two", "POL-1092"),
    ],
)
def test_spoken_identifiers_are_canonicalised_before_the_agent_sees_them(
    spoken: str, expected: str
) -> None:
    """STT never returns POL-1092. Pydantic rejects "policy ten ninety two",
    the agent asks the caller to repeat themselves, and the call feels broken."""
    assert expected in normalize_transcript(spoken)


def test_an_already_canonical_id_is_not_duplicated() -> None:
    assert normalize_transcript("check claim CLM-8821").count("CLM-8821") == 1


def test_a_transcript_with_no_identifier_is_untouched() -> None:
    text = "what does my policy cover for water damage"
    assert normalize_transcript(text) == text


# ---------------------------------------------------- screen vs ear split

@pytest.mark.parametrize(
    ("event_type", "payload", "shown"),
    [
        ("queued", {"position": 3}, True),
        ("queued", {"position": 0}, False),   # no queue, nothing to say
        ("tool_start", {"name": "get_claim_status"}, True),
        ("token", {"text": "hello"}, False),  # spoken, not shown
        ("done", {}, True),
    ],
)
def test_browser_mirror_selects_the_right_events(event_type, payload, shown) -> None:
    event = RunEvent(run_id="r", type=event_type, seq=0, payload=payload)
    assert (browser_message(event) is not None) is shown


# ------------------------------------------------------------------ helpers

def _ctx(text: str):
    from livekit.agents import llm

    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


def _conn():
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

    return DEFAULT_API_CONNECT_OPTIONS


async def _drain(stream) -> str:
    """Collect everything the stream would speak."""
    parts: list[str] = []
    async for chunk in stream:
        content = getattr(chunk.delta, "content", None) if chunk.delta else None
        if content:
            parts.append(content)
    return " ".join(parts)


# ------------------------------------------------------------ configuration

BASE_CONFIG = {
    "stt_api_key": "k", "tts_api_key": "k",
    "livekit_url": "ws://x", "tts_provider": "piper",
}


@pytest.mark.parametrize(
    ("missing", "expected_fragment"),
    [
        ("stt_api_key", "speech recognition"),
        ("livekit_url", "cannot register"),
    ],
)
def test_missing_credentials_are_reported_at_startup(missing, expected_fragment) -> None:
    """A missing key first surfaced from inside a plugin when a caller joined a
    room - after the worker had registered and looked healthy. An operator
    should learn about it before anyone dials in."""
    from voice.app.settings import VoiceSettings
    from voice.app.worker import check_config

    problems = check_config(VoiceSettings(**{**BASE_CONFIG, missing: ""}))
    assert any(expected_fragment in p for p in problems), problems


def test_a_missing_tts_key_matters_only_for_the_hosted_provider() -> None:
    """Local synthesis needs no key at all - that is the point of making it the
    default. Reporting a missing one would be noise."""
    from voice.app.settings import VoiceSettings
    from voice.app.worker import check_config

    local = check_config(VoiceSettings(**{**BASE_CONFIG, "tts_api_key": ""}))
    assert not any("synthesis" in p for p in local)

    hosted = check_config(
        VoiceSettings(**{**BASE_CONFIG, "tts_api_key": "", "tts_provider": "groq"})
    )
    assert any("synthesis" in p for p in hosted)


def test_a_fully_configured_worker_reports_no_problems() -> None:
    from voice.app.settings import VoiceSettings
    from voice.app.worker import check_config

    assert check_config(VoiceSettings(**BASE_CONFIG)) == []
