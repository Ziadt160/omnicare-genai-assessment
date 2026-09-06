"""Graph assembly and routing.

    guard ──blocked───────────────────────────────────────────────┐
      │ ok                                                       │
      ▼                                                          │
    agent ──no tool call──► readback ──────────────────────────┐  │
      │  │                                                     │  │
      │  ├─read tool──► tools ──────────┐                      │  │
      │  └─write tool─► capture ─► confirm ──declined──────────┤  │
      │                              │ approved                │  │
      ◄──────────────────────────────┴─────────────────────────┘  │
                                                                  ▼
                                            parse ─► ground ─► format ─► END

`tools` returns to `agent` so the model can read the result; `confirm` sits
between the model's decision to write and the write itself.

Every exit runs `parse ─► ground ─► format`, including a blocked turn, so the
response shape is identical however the turn ended and the gateway never
special-cases a refusal. The order inside that tail is a guarantee, not a
convention: `parse` unwraps a structured reply before `ground` edits prose (a
JSON envelope is not prose, and `ground` cut a span out of the middle of one),
and `format` is last because it is the only node that should touch the text the
reader finally sees.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from libs.contracts import Chunk, ToolCall
from libs.guardrails.injection import POLICY_MARKER, TOOL_MARKER, as_data
from .nodes import (
    WRITE_TOOLS,
    make_agent_node,
    make_confirm_node,
    make_format_node,
    make_parse_node,
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
                # "ok" means the tool did what it was asked, not merely that it
                # returned. A tool that declines - a claim above the policy's
                # cap, a claim id that does not exist - reports an `error` key,
                # and showing that as a green chip tells the policyholder their
                # claim was filed when it was refused.
                status = (
                    "error"
                    if isinstance(result, dict) and result.get("error")
                    else "ok"
                )
            except Exception as exc:  # surfaced to the model, not raised
                result = {"error": type(exc).__name__, "detail": str(exc)}
                status = "error"

            # A payment split is read straight out of the policy without going
            # through search, so nothing was adding it to `retrieved` and
            # `ground` had no citation to report - the answer named a section
            # and the UI showed none. The section it actually used is recorded
            # here, in the same shape a searched chunk takes, so one grounding
            # rule covers both routes.
            if name == "estimate_claim_payment" and isinstance(result, dict):
                citation = result.get("citation")
                if citation:
                    retrieved.append(
                        Chunk(
                            chunk_id=citation,
                            # The sentences the figures were read from - the
                            # part of the section that actually informed the
                            # answer, which is what a citation should point at.
                            text="\n".join(result.get("policy_says") or []),
                            source_file=citation.split(" § ")[0],
                            section_id="",
                            section_title=result.get("section", ""),
                            char_start=0,
                            char_end=0,
                            citation=citation,
                        ).model_dump(mode="json")
                    )

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
            # Marked as data, not conversation. Retrieved policy text gets the
            # `<policy_document>` marker the system prompt already named; every
            # other tool result gets `<tool_result>`. Soft control - a model can
            # be talked past a delimiter - but until now the prompt described a
            # boundary that nothing drew.
            marker = (
                POLICY_MARKER if name == "search_policy_documents" else TOOL_MARKER
            )
            outputs.append(
                ToolMessage(
                    content=as_data(json.dumps(result, default=str), marker=marker),
                    tool_call_id=call_id,
                )
            )

        return {
            "messages": outputs,
            "tool_invocations": invocations,
            "retrieved": retrieved,
            "pending_write": None,
        }

    return tools_node


def make_capture_node(settlement: Any | None = None):
    """Lift the write arguments out of the model's tool call so `confirm` can
    read them back before anything is executed - and work out, from the policy
    document, what the claim would actually pay.

    The split is computed here rather than in `confirm` because `confirm` calls
    `interrupt()`, which suspends and later re-enters the node: anything it
    computed before interrupting is computed again on resume, and an await on
    the retrieval service is not something to repeat on the way back. `capture`
    runs once, and what it leaves in state survives the suspension through the
    checkpointer.

    Args:
        settlement: Optional async ``(claim_type, amount) -> Settlement|None``.
            Omitted, the confirmation reads back as it always did - a
            confirmation prompt that works is not worth breaking over a
            breakdown that could not be computed.
    """

    async def capture(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        for call in getattr(last, "tool_calls", []) or []:
            if call["name"] not in WRITE_TOOLS:
                continue
            args = call["args"]
            update: dict[str, Any] = {"pending_write": args, "pending_settlement": None}
            if settlement is None:
                return update
            try:
                split = await settlement(args.get("claim_type"), args.get("amount"))
            except Exception:
                # A malformed amount reaches here as whatever the model typed;
                # Pydantic rejects it inside the tool, with a message the model
                # can recover from. Losing the breakdown is the right cost -
                # failing the confirmation gate over it would turn a bad
                # argument into a lost write.
                split = None
            if split is not None:
                update["pending_settlement"] = {
                    **split.as_dict(),
                    "payment_summary": split.summary(),
                }
                # And say where the figures came from.
                #
                # Reported, and a fair question: the confirmation states a
                # $25,000 limit and a $500 deductible with no citation, so how
                # does it know them without looking? It did look - `settlement`
                # reads the policy document straight from the retrieval
                # service, which is why the numbers are right - but that path
                # never touched `retrieved`, so nothing downstream could name
                # the section and the reader was asked to take two figures on
                # trust at the exact moment they were being asked to approve a
                # claim.
                #
                # `sources` is set here as well as `retrieved` because `ground`
                # does not run on this turn: the graph suspends at `confirm`,
                # so there is nothing between here and the policyholder. That
                # is safe in a way a model's citation would not be - this
                # citation is built by `Settlement.citation` from the section
                # the figures were parsed out of, and comes back empty rather
                # than guessed when the source file is unknown.
                citation = split.citation
                if citation:
                    update["retrieved"] = [
                        *(state.get("retrieved") or []),
                        Chunk(
                            chunk_id=citation,
                            text="\n".join(split.policy_says),
                            source_file=split.source_file,
                            section_id="",
                            section_title=split.section_title,
                            char_start=0,
                            char_end=0,
                            citation=citation,
                        ).model_dump(mode="json"),
                    ]
                    update["sources"] = [citation]
            return update
        return {}

    return capture


def build_graph(
    llm: Any,
    tools: list[Any],
    *,
    checkpointer: Any | None = None,
    require_confirmation: bool = True,
    max_iterations: int = 5,
    settlement: Any | None = None,
):
    g = StateGraph(AgentState)

    g.add_node("guard", make_guard_node())
    g.add_node("agent", make_agent_node(llm, tools, max_iterations=max_iterations))
    g.add_node("capture", make_capture_node(settlement))
    g.add_node("tools", make_tools_node(tools))
    g.add_node("confirm", make_confirm_node(require_confirmation))
    g.add_node("readback", make_readback_node())
    g.add_node("parse", make_parse_node())
    g.add_node("ground", make_ground_node())
    g.add_node("format", make_format_node())

    g.set_entry_point("guard")
    g.add_conditional_edges("guard", route_after_guard, {"blocked": "parse", "ok": "agent"})
    g.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "confirm": "capture", "done": "readback"},
    )
    g.add_edge("readback", "parse")
    g.add_edge("capture", "confirm")
    g.add_conditional_edges("confirm", route_after_confirm, {"tools": "tools", "done": "parse"})
    g.add_edge("tools", "agent")
    # `format` is last: `ground` may rewrite the answer, and normalising a
    # text that is about to change would normalise the wrong one.
    g.add_edge("parse", "ground")
    g.add_edge("ground", "format")
    g.add_edge("format", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
