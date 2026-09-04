"""Events written to ``stream:{run_id}`` and forwarded over WebSocket / voice.

One event vocabulary serves all three surfaces: REST buffers them into a
``ChatResponse``, WebSocket forwards each as it arrives, and the voice worker
feeds ``token`` events to TTS.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunEventType = Literal[
    "queued",       # accepted; payload carries queue position
    "started",      # a worker picked it up
    "token",        # one chunk of assistant text
    "tool_start",   # a tool call began - voice uses this to speak a filler
    "tool_end",
    "confirm",      # interrupt(): payload is the readback + pending args
    "sources",      # verified citations, emitted by the ground node
    "done",         # terminal
    "error",        # terminal
]


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    type: RunEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    seq: int = Field(ge=0)

    @property
    def is_terminal(self) -> bool:
        return self.type in ("done", "error")
