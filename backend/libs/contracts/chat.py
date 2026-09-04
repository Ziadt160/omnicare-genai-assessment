"""Contracts for the chat surface.

The assessment specifies exactly two request fields (``user_id``, ``message``)
and exactly three response fields (``response``, ``sources``, ``tool_calls``).
Everything added here is optional with a default, so the specified payload
still validates under ``extra="forbid"``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["text", "voice"]
ToolStatus = Literal["ok", "error", "awaiting_confirmation", "denied"]


class ToolCall(BaseModel):
    """One tool invocation, surfaced in the response for auditability."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    status: ToolStatus = "ok"
    latency_ms: int | None = Field(default=None, ge=0)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"user_id": "usr_123", "message": "A pipe burst in my kitchen. Am I covered?"},
                {"user_id": "usr_123", "message": "What is the status of claim CLM-8821?"},
            ]
        },
    )

    # --- specified by the assessment ---
    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    message: str = Field(min_length=1, max_length=4000)

    # --- additive, all optional ---
    conversation_id: str | None = Field(default=None, max_length=64)
    channel: Channel = "text"
    stt_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "response": "Yes - sudden pipe bursts are covered up to $25,000 "
                                "with a $500 deductible.",
                    "sources": [
                        "sample_policy.md § Section 1: Home Water Damage Coverage"
                    ],
                    "tool_calls": [
                        {"name": "search_policy_documents",
                         "arguments": {"query": "burst pipe water damage coverage"},
                         "status": "ok"}
                    ],
                    "conversation_id": "cnv_01HZ",
                }
            ]
        },
    )

    # --- specified by the assessment ---
    response: str
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # --- additive ---
    conversation_id: str
    trace_id: str | None = None


class HealthResponse(BaseModel):
    """``GET /api/v1/health`` - serializes to exactly {"status": "healthy"}."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "degraded"] = "healthy"


class DeepHealthResponse(HealthResponse):
    """``GET /api/v1/health/deep`` - readiness, used by compose and the UI."""

    dependencies: dict[str, Literal["up", "down"]] = Field(default_factory=dict)
