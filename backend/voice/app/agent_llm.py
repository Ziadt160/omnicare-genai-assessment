"""The OmniCare agent, presented to LiveKit as an ``llm.LLM``.

This is the seam that keeps the voice channel honest. LiveKit's `AgentSession`
wants something it can call for a completion; what it gets is a shim that
enqueues onto the same `jobs:chat` stream the gateway uses and streams the
agent's tokens back.

The alternative - giving the voice worker its own prompt, its own tools and its
own model client - is the obvious shape and the wrong one. Two implementations
of the same assistant drift within days, the guardrails have to be duplicated,
the confirmation flow has to be re-implemented, and every eval has to be run
twice against two things that are no longer the same. Here the voice path
inherits injection screening, the bounded loop, `interrupt()` before a write
and citation grounding because it is literally the same graph.

What voice adds, and text does not need:

* ``channel="voice"`` on the request, which is what selects the confirmation
  tier and the phonetic read-back in the graph.
* A spoken filler the moment a tool starts, because several seconds of silence
  on a phone call reads as a dropped connection rather than as thinking.
* Citations pushed to the browser over the data channel instead of read aloud.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, llm

from libs.contracts import RunEvent

log = logging.getLogger("omnicare.voice.llm")

# Spoken as soon as a tool call begins. Without it the pause while retrieval or
# a claim lookup runs is indistinguishable from a dead line.
TOOL_FILLERS = {
    "search_policy_documents": "Let me check your policy.",
    "get_claim_status": "Let me pull that claim up.",
    "submit_claim": "One moment.",
}


class OmniCareLLM(llm.LLM):
    """Routes a turn through the agent service rather than a model.

    Args:
        queue: The same ``JobQueue`` the gateway enqueues onto.
        user_id: LiveKit participant identity - becomes the policyholder id.
        conversation_id: The room name, so a voice conversation has the same
            durable history as a chat one and can resume a paused confirmation.
        on_event: Called for every run event, so the worker can mirror
            transcripts and citations to the browser over the data channel.
    """

    def __init__(
        self,
        queue: Any,
        *,
        user_id: str,
        conversation_id: str,
        timeout_s: float = 30.0,
        on_event: Callable[[RunEvent], Any] | None = None,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._timeout_s = timeout_s
        self._on_event = on_event
        self.last_confidence: float | None = None

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        # `tools` is deliberately ignored. The agent owns tool selection and
        # execution; exposing them here would put a second, competing tool loop
        # inside the voice worker.
        return _AgentStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


def _latest_user_text(chat_ctx: llm.ChatContext) -> str:
    """The most recent user turn - LiveKit hands over the whole context."""
    for item in reversed(list(chat_ctx.items)):
        if getattr(item, "role", None) == "user":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                parts = [c for c in content if isinstance(c, str)]
                if parts:
                    return " ".join(parts).strip()
            elif isinstance(content, str):
                return content.strip()
    return ""


class _AgentStream(llm.LLMStream):
    """Streams one agent run back to the session as chat chunks."""

    async def _run(self) -> None:
        shim: OmniCareLLM = self._llm  # type: ignore[assignment]
        message = _latest_user_text(self._chat_ctx)

        if not message:
            # No user turn to answer. Reached when something asks the session
            # to speak without a question - the greeting used to land here and
            # be answered "Sorry, I didn't catch that", which is a strange way
            # to open a call. The greeting is now spoken directly; this remains
            # for a genuinely empty transcript.
            log.debug("no user turn in chat context; nothing to ask the agent")
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=uuid.uuid4().hex,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content="Sorry, I didn't catch that. Could you say it again?",
                    ),
                )
            )
            return

        run_id = uuid.uuid4().hex
        await shim._queue.enqueue(
            run_id,
            {
                "user_id": shim._user_id,
                "conversation_id": shim._conversation_id,
                "message": message,
                # The one field that changes the graph's behaviour: it selects
                # the confirmation tier and the phonetic read-back.
                "channel": "voice",
                "stt_confidence": shim.last_confidence,
            },
            deadline_s=shim._timeout_s,
        )

        spoke = False
        async for event in shim._queue.subscribe(run_id, shim._timeout_s):
            if shim._on_event is not None:
                await shim._on_event(event)

            text = self._spoken(event)
            if text:
                spoke = True
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=run_id,
                        delta=llm.ChoiceDelta(role="assistant", content=text),
                    )
                )

        if not spoke:
            # Never leave the caller in silence. A turn that produced nothing
            # sayable is still a turn they are waiting on.
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=run_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content="Sorry, I wasn't able to complete that. "
                        "Could you try again?",
                    ),
                )
            )

    @staticmethod
    def _spoken(event: RunEvent) -> str:
        """What, if anything, should be said aloud for this event.

        Returning "" means the event is for the screen only. Queue positions and
        tool chips belong in the browser, not in the caller's ear; citations are
        rendered visually because reading "sample_policy.md section 1" aloud is
        noise.
        """
        if event.type == "tool_start":
            return TOOL_FILLERS.get(str(event.payload.get("name", "")), "One moment.")
        if event.type == "token":
            return str(event.payload.get("text", ""))
        if event.type == "confirm":
            # The read-back is generated in the graph, not here - see ADR 0007.
            return str(event.payload.get("readback", ""))
        if event.type == "error":
            return "Sorry, something went wrong on my end. Could you try again?"
        return ""
