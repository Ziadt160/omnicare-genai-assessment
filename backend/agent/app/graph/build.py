"""Graph assembly and routing.

    guard ──blocked───────────────────────────────────────► ground ─► END
      │ ok
      ▼
    agent ◄──┬── tools ◄── confirm (writes only)
      │      │
      │      └── tools (reads)
      ▼ done
    readback ─► ground ─► END

`tools` returns to `agent` so the model can read the result; `confirm` sits
between the model's decision to write and the write itself.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from libs.contracts import Chunk, ToolCall
from .nodes import (
    WRITE_TOOLS,
    make_agent_node,
    make_confirm_node,
    make_ground_node,
    make_guard_node,
    make_readback_node,
)
from .state import AgentState


def route_after_guard(state: AgentState) -> Literal["blocked", "ok"]:
    return "blocked" if state.get("guard_blocked") else "ok"


def route_after_agent(state: AgentState) -> Literal["tools", "confirm", "done"]:
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "done"
    if any(c["name"] in WRITE_TOOLS for c in calls):
        return "confirm"
    return "tools"


def route_after_confirm(state: AgentState) -> Literal["tools", "done"]:
    return "tools" if state.get("pending_write") else "done"


def make_tools_node(tools: list[Any]):
    """Execute tool calls, recording each one and harvesting retrieved chunks.

    A validation failure returns a structured ToolMessage the model can recover
    from rather than raising - an agent that crashes on a malformed argument is
    worse than one that is told the argument was malformed.
    """
    by_name = {t.name: t for t in tools}

    async def tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last = messages[-1]
        outputs: list[ToolMessage] = []
        invocations: list[dict[str, Any]] = list(state.get("tool_invocations", []))
        retrieved: list[dict[str, Any]] = list(state.get("retrieved", []) or [])

        for call in getattr(last, "tool_calls", []) or []:
            name, args, call_id = call["name"], call["args"], call["id"]
            tool = by_name.get(name)
            if tool is None:
                outputs.append(
                    ToolMessage(content=f"Unknown tool {name}", tool_call_id=call_id)
                )
                continue

            try:
                result = await tool.ainvoke(args)
                status = "ok"
            except Exception as exc:  # surfaced to the model, not raised
                result = {"error": type(exc).__name__, "detail": str(exc)}
                status = "error"

            if name == "search_policy_documents" and isinstance(result, dict):
                for raw in result.get("chunks", []):
                    # Validated through Chunk so the citation is filled and the
                    # shape is checked, then stored as a plain dict - see the
                    # note on AgentState.
                    retrieved.append(
                        Chunk(
                            chunk_id=raw.get("citation", ""),
                            text=raw["text"],
                            source_file=raw["source_file"],
                            section_id=raw.get("section_id", ""),
                            section_title=raw["section_title"],
                            char_start=0,
                            char_end=len(raw["text"]),
                            score=raw.get("score"),
                            citation=raw.get("citation", ""),
                        ).model_dump(mode="json")
                    )

            invocations.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    result=result if isinstance(result, dict) else {"value": str(result)},
                    status=status,  # type: ignore[arg-type]
                ).model_dump(mode="json")
            )
            outputs.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call_id)
            )

        return {
            "messages": outputs,
            "tool_invocations": invocations,
            "retrieved": retrieved,
            "pending_write": None,
        }

    return tools_node


def capture_pending_write(state: AgentState) -> dict[str, Any]:
    """Lift the write arguments out of the model's tool call so `confirm` can
    read them back before anything is executed."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    for call in getattr(last, "tool_calls", []) or []:
        if call["name"] in WRITE_TOOLS:
            return {"pending_write": call["args"]}
    return {}


def build_graph(
    llm: Any,
    tools: list[Any],
    *,
    checkpointer: Any | None = None,
    require_confirmation: bool = True,
    max_iterations: int = 5,
):
    g = StateGraph(AgentState)

    g.add_node("guard", make_guard_node())
    g.add_node("agent", make_agent_node(llm, tools, max_iterations=max_iterations))
    g.add_node("capture", capture_pending_write)
    g.add_node("tools", make_tools_node(tools))
    g.add_node("confirm", make_confirm_node(require_confirmation))
    g.add_node("readback", make_readback_node())
    g.add_node("ground", make_ground_node())

    g.set_entry_point("guard")
    g.add_conditional_edges("guard", route_after_guard, {"blocked": "ground", "ok": "agent"})
    g.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "confirm": "capture", "done": "readback"},
    )
    g.add_edge("readback", "ground")
    g.add_edge("capture", "confirm")
    g.add_conditional_edges("confirm", route_after_confirm, {"tools": "tools", "done": "ground"})
    g.add_edge("tools", "agent")
    g.add_edge("ground", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
