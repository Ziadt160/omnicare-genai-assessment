"""The LiveKit voice worker.

A dispatch worker, not an HTTP server: it registers with the SFU and waits to be
assigned to rooms. That different runtime model, plus the audio and ONNX
dependencies, is why it is a separate service from the agent.

The rule this file is built around: **it does not reimplement the assistant.**
It supplies ears and a mouth; `OmniCareLLM` routes every turn through the same
`jobs:chat` queue the gateway uses, so voice inherits injection screening, the
bounded loop, `interrupt()` before a write and citation grounding for free. One
graph, one eval suite, two surfaces.

Written against livekit-agents **1.7**, whose `AgentSession` replaced the manual
VAD/STT/TTS wiring of the 0.12 line. The two are not variations of one API, and
this file was first drafted against the older one from memory - the version is
pinned in the Dockerfile and verified against the installed package.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from libs.contracts import RunEvent, conversation_id_from_room
from libs.guardrails.normalize import normalize_claim_id, normalize_policy_number
from .settings import VoiceSettings

log = logging.getLogger("omnicare.voice")

# Spoken verbatim when a caller joins. Fixed text rather than a generated
# greeting: it costs nothing, it never drifts, and it cannot hallucinate a
# capability on the first sentence of the call.
GREETING = (
    "Hello, you've reached the OmniCare policyholder assistant. "
    "I can check what your policy covers, look up a claim, or help you file "
    "one. How can I help?"
)

INSTRUCTIONS = """You are the OmniCare Financial policyholder assistant, \
speaking with a policyholder by voice.

Everything you say comes from the assistant service. Relay it faithfully and \
concisely. Do not add coverage details, claim statuses or policy numbers of \
your own - if the service did not say it, you do not know it."""


def normalize_transcript(text: str) -> str:
    """Rewrite spoken identifiers into canonical form before the agent sees them.

    STT returns "policy ten ninety two", never "POL-1092". Pydantic rejects the
    former, the agent asks the caller to repeat themselves, and the call feels
    broken - so normalization happens at the point of transcription rather than
    being left for the model to guess at.
    """
    claim = normalize_claim_id(text)
    policy = normalize_policy_number(text)

    rewritten = text
    if claim and claim not in rewritten:
        rewritten = f"{rewritten} (claim {claim})"
    if policy and policy not in rewritten:
        rewritten = f"{rewritten} (policy {policy})"
    return rewritten


def browser_message(event: RunEvent) -> dict[str, Any] | None:
    """The screen half of an event, or None if there is nothing to show.

    Spoken and shown are not alternatives. Citations are shown and never spoken -
    reading "sample_policy.md section 1" aloud is noise - and queue position and
    tool activity are the same. But the answer itself is **both**: a call used to
    leave no readable record of what the assistant said, so hanging up left
    nothing to re-read and a policyholder who misheard a deductible had no way to
    check it.
    """
    if event.type == "queued":
        position = int(event.payload.get("position", 0) or 0)
        if not position:
            return None
        return {"type": "state", "label": f"queued · {position} ahead", "kind": "warn"}
    if event.type == "tool_start":
        return {"type": "state", "label": str(event.payload.get("name", "working")),
                "kind": "busy"}
    if event.type == "tool_end":
        return {"type": "tool", "name": str(event.payload.get("name", "")),
                "status": str(event.payload.get("status", "ok"))}
    if event.type == "token":
        # The same text the TTS is speaking, streamed to the transcript so the
        # two stay in step. Empty deltas are dropped rather than published.
        text = str(event.payload.get("text", ""))
        return {"type": "answer_delta", "text": text} if text else None
    if event.type == "sources":
        # Deliberately carries no `text`. This was published as an `answer` with
        # an empty string, which the browser rendered as a *new* assistant
        # bubble - detaching the citations from the answer they belonged to.
        sources = list(event.payload.get("sources", []))
        return {"type": "sources", "sources": sources} if sources else None
    if event.type == "confirm":
        return {"type": "confirm", "readback": str(event.payload.get("readback", "")),
                "args": event.payload.get("args", {})}
    if event.type == "done":
        return {"type": "state", "label": "listening", "kind": "ok"}
    return None


def check_config(settings: VoiceSettings) -> list[str]:
    """Configuration problems that would otherwise surface per-call.

    A missing key raised from inside the TTS plugin the first time a caller
    joins a room - after the worker had registered and reported itself healthy.
    Checking at startup turns that into a log line an operator can act on
    before anyone dials in.
    """
    problems: list[str] = []
    if not settings.stt_api_key:
        problems.append("VOICE_STT_API_KEY is empty - speech recognition will fail")
    if settings.tts_provider == "groq" and not settings.tts_api_key:
        problems.append("VOICE_TTS_API_KEY is empty - speech synthesis will fail")
    if not settings.livekit_url:
        problems.append("VOICE_LIVEKIT_URL is empty - the worker cannot register")
    return problems


def check_speech(settings: VoiceSettings) -> list[str]:
    """Actually call the speech endpoint once at startup.

    Config presence is not the same as config correctness: the key was set, the
    model id came from the live catalogue, and synthesis still failed - once
    because the plugin sent a field Groq rejects, once because the model needs
    one-time terms acceptance. Both surfaced only when a caller was already in
    the room. One request at boot moves that to a log line.
    """
    if settings.tts_provider == "piper":
        from .piper_tts import probe

        ok, detail = probe(settings.piper_model_path, settings.piper_config_path or None)
        label = f"piper:{settings.piper_model_path}"
    else:
        from .groq_tts import probe as groq_probe

        ok, detail = groq_probe(
            settings.tts_model, settings.tts_voice,
            settings.tts_api_key, settings.tts_base_url,
        )
        label = settings.tts_model

    if ok:
        log.info("speech synthesis ready (%s): %s", label, detail)
        return []
    return [f"speech synthesis unavailable ({label}): {detail}"]


def build_session(settings: VoiceSettings, queue: Any, *, user_id: str,
                  conversation_id: str, on_event: Any = None):
    """Assemble the pipeline. Separated from `entrypoint` so it is inspectable
    without a LiveKit server."""
    from livekit.agents import AgentSession
    from livekit.plugins import openai, silero

    from .agent_llm import OmniCareLLM
    from .piper_tts import build as build_tts

    shim = OmniCareLLM(
        queue,
        user_id=user_id,
        conversation_id=conversation_id,
        timeout_s=settings.run_timeout_s,
        on_event=on_event,
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(
            model=settings.stt_model,
            base_url=settings.stt_base_url,
            api_key=settings.stt_api_key,
        ),
        llm=shim,
        # Never openai.TTS: the plugin always sends `stream_format`, which Groq
        # rejects with a 400. Piper by default, Groq when explicitly asked for.
        tts=build_tts(settings),
    )
    return session, shim


async def entrypoint(ctx: Any) -> None:
    """Serve one room."""
    from livekit.agents import Agent

    # `livekit.agents.room_io` is a re-exported attribute, not an importable
    # module path - `from livekit.agents.room_io import ...` raises
    # ModuleNotFoundError. The real location is livekit.agents.voice.room_io.
    from livekit.agents.voice.room_io import RoomOptions

    from libs.adapters.queue_redis import RedisQueue

    settings = VoiceSettings()
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    user_id = participant.identity or "voice-user"
    # The room name is the only thing a dispatched worker is told, so the
    # gateway puts the conversation id inside it. Unwrapping it here is what
    # makes a call and a typed conversation one thread rather than two.
    conversation_id = conversation_id_from_room(ctx.room.name)
    log.info("serving room %s as conversation %s for %s",
             ctx.room.name, conversation_id, user_id)

    queue = RedisQueue(settings.redis_url)

    async def publish(payload: dict[str, Any]) -> None:
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode(), reliable=True
            )
        except Exception as exc:  # a closed data channel must not kill the call
            log.debug("data channel publish failed: %s", exc)

    async def on_event(event: RunEvent) -> None:
        message = browser_message(event)
        if message is not None:
            await publish(message)

    session, shim = build_session(
        settings, queue,
        user_id=user_id, conversation_id=conversation_id, on_event=on_event,
    )

    @session.on("user_input_transcribed")
    def _on_transcript(event: Any) -> None:
        """Mirror what was heard to the screen, and carry STT confidence.

        Visual confirmation of the transcript costs nothing and consumes no
        conversational turn - the cheapest half of ADR 0007.
        """
        text = getattr(event, "transcript", "") or ""
        is_final = bool(getattr(event, "is_final", False))
        confidence = getattr(event, "confidence", None)
        if confidence is not None:
            shim.last_confidence = float(confidence)

        import asyncio

        asyncio.create_task(
            publish({
                "type": "transcript_final" if is_final else "transcript_partial",
                "text": text,
            })
        )

    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
        room=ctx.room,
        room_options=RoomOptions(),
    )

    # `say`, not `generate_reply`. The greeting is fixed text, so routing it
    # through the agent would spend a graph run and a model call to produce a
    # sentence already written here - and the shim, finding no user turn to
    # answer, would open the call with "Sorry, I didn't catch that".
    await session.say(GREETING)


def main() -> None:
    from livekit.agents import WorkerOptions, cli

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    settings = VoiceSettings()

    for problem in check_config(settings):
        log.error("voice worker misconfigured: %s", problem)
    if settings.verify_speech_on_start:
        for problem in check_speech(settings):
            log.error("voice worker: %s", problem)

    # Passed explicitly rather than left to livekit-agents' own LIVEKIT_URL /
    # LIVEKIT_API_KEY / LIVEKIT_API_SECRET lookup. Every other service in this
    # system reads a prefixed settings class, and having one of them silently
    # depend on bare environment names is the kind of inconsistency that only
    # shows up as "ws_url is required" at startup - which is exactly how this
    # was found.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )


if __name__ == "__main__":
    main()
