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
    expires_in: int = Field(default=900, ge=60, le=3600)
