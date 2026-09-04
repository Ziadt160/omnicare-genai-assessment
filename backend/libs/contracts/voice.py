"""Contracts for LiveKit access.

The API secret never leaves the gateway; the browser receives only a
short-lived JWT scoped to a single room.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VoiceTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    conversation_id: str | None = Field(default=None, max_length=64)


class VoiceTokenResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    url: str = Field(examples=["ws://localhost:7880"])
    room: str
    # Returned even when the caller supplied one, because the caller may not
    # have: a browser whose first action is pressing the mic has no conversation
    # yet, and it needs this id so the typed turns that follow land on the same
    # thread the call used.
    conversation_id: str = Field(examples=["cnv_9f2a41b7c0de"])
    expires_in: int = Field(default=900, ge=60, le=3600)


# The room name is the only thing the voice worker is told when it is dispatched
# - no headers, no body, just a room. So the conversation id travels inside it,
# and both ends of that trip live here rather than as a format string in the
# gateway and a `lstrip` in the worker.
ROOM_PREFIX = "omnicare-"


def room_for(conversation_id: str) -> str:
    """The LiveKit room carrying a given conversation."""
    return f"{ROOM_PREFIX}{conversation_id}"


def conversation_id_from_room(room: str) -> str:
    """Recover the conversation id the room was created for.

    Uses ``removeprefix`` rather than splitting on the separator: Postgres mints
    UUIDs, which contain hyphens, and a `split("-")[1]` would truncate
    ``omnicare-550e8400-e29b-...`` to ``550e8400`` - giving every call its own
    memory while looking like it worked.

    A room that does not carry the prefix is returned unchanged, so a room
    created by hand is still usable as a thread id instead of raising mid-call.
    """
    return room.removeprefix(ROOM_PREFIX)
