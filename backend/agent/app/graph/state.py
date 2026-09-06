"""Graph state.

The LLM writes to `messages` only. Everything else is written by deterministic
nodes, which is what lets the guarantees in docs/adr/0002 hold: the model
chooses tools, it does not decide whether input was safe or whether a citation
was real.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from libs.contracts import Channel


class AgentState(TypedDict, total=False):
    # --- conversation ---
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    conversation_id: str
    channel: Channel
    stt_confidence: float | None

    # --- retrieval, written by the tools node ---
    #
    # Plain dicts, not Chunk/ToolCall. Everything in graph state is serialized
    # into the checkpoint, and LangGraph warns that deserializing unregistered
    # types "will be blocked in a future version" - so storing Pydantic models
    # here is a working system with a scheduled expiry date. The models still
    # own validation at the edges; the state carries their `model_dump()` and
    # rehydrates on read.
    retrieved: list[dict[str, Any]]

    # --- outputs, written by `ground` ---
    sources: list[str]

    # What the model claimed about its own answer, and what the system was
    # willing to stand behind. Kept apart deliberately: a model reporting 0.95
    # on an answer with no retrieval behind it is the case worth seeing, and
    # storing only the final number would hide exactly that.
    answer_confidence: float | None
    model_confidence: float | None
    answer_unknown: bool
    confidence_reason: str | None
    tool_invocations: list[dict[str, Any]]

    # --- control ---
    guard_blocked: bool
    guard_rule: str | None
    guard_flagged: bool
    pending_write: dict[str, Any] | None
    # The payment split for the write awaiting confirmation, computed by
    # `capture` from the policy document. Shown to the policyholder in the
    # confirmation prompt: agreeing to file a $35,000 claim means something
    # different once you can see that $10,000 of it lands on you.
    pending_settlement: dict[str, Any] | None
    confirmation_tier: Literal[0, 1, 2]
    iterations: int
    stopped_reason: str | None
