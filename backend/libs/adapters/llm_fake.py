"""A scripted chat model. The single most important object in the test suite.

You cannot practise TDD against a real LLM: it is slow, non-deterministic,
rate-limited, and costs quota on every red-green cycle. ``FakeLLM`` replaces it
with a script - a list of turns the model will "decide" on, in order - so the
whole graph above it becomes deterministic and instant.

Written first, it shapes the interfaces correctly. Written last, it is a
retrofit that only tests what the code already happens to do.

    llm = FakeLLM([
        ToolTurn("get_claim_status", {"claim_id": "CLM-8821"}),
        TextTurn("Claim CLM-8821 is Approved."),
    ])

Every call is recorded on ``llm.calls`` so a test can assert what the graph
actually sent - the system prompt, the tool schemas, the message history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


@dataclass
class ToolTurn:
    """The model decides to call one tool."""

    name: str
    args: dict[str, Any]
    text: str = ""


@dataclass
class TextTurn:
    """The model answers in prose and stops."""

    text: str


@dataclass
class ErrorTurn:
    """The model call raises - used to exercise retry and circuit breaking."""

    exc: Exception


Turn = ToolTurn | TextTurn | ErrorTurn


@dataclass
class RecordedCall:
    messages: list[BaseMessage]
    tools: list[str] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        for m in self.messages:
            if m.type == "system":
                return str(m.content)
        return ""

    @property
    def last_human(self) -> str:
        for m in reversed(self.messages):
            if m.type == "human":
                return str(m.content)
        return ""


class FakeLLM(BaseChatModel):
    """A ``BaseChatModel`` that returns a fixed script.

    Args:
        script: Turns to return, in order. When the script is exhausted the
            model repeats the final turn rather than raising, so a graph that
            loops more than expected fails on the *assertion* about iteration
            count rather than on a confusing StopIteration.
    """

    script: list[Turn] = []
    calls: list[RecordedCall] = []
    bound_tools: list[str] = []

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, script: Sequence[Turn] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.script = list(script or [TextTurn("(no script)")])
        self.calls = []
        self.bound_tools = []

    # -- BaseChatModel plumbing -------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(
        self, tools: Sequence[dict[str, Any] | type | BaseTool], **kwargs: Any
    ) -> Runnable:
        self.bound_tools = [
            t.name if isinstance(t, BaseTool) else getattr(t, "__name__", str(t))
            for t in tools
        ]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(len(self.calls), len(self.script) - 1)
        self.calls.append(RecordedCall(messages=list(messages), tools=list(self.bound_tools)))
        turn = self.script[index]

        if isinstance(turn, ErrorTurn):
            raise turn.exc

        if isinstance(turn, ToolTurn):
            message = AIMessage(
                content=turn.text,
                tool_calls=[
                    {
                        "name": turn.name,
                        "args": turn.args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content=turn.text)

        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- test helpers ------------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def tool_names_offered(self) -> list[str]:
        """Which tools the graph actually bound - guards against a silent
        regression where a tool stops being registered and the model can no
        longer choose it."""
        return self.bound_tools
