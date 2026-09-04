"""A keyless chat model for demos and end-to-end tests.

Selected with ``LLM_PROVIDER=fake``. **Never the default**, and it says so in
every answer it produces.

Why it exists: without it, nothing about this system can be exercised end to
end without a credential. With it, `docker compose up` gives a reviewer a
working chat - real queue, real consumer group, real graph, real retrieval,
real guardrails, real confirmation flow, real Postgres history - and only the
model's judgement is substituted. That is also exactly the boundary the
`tests/e2e` suite wants: it should fail when the plumbing breaks, not when a
free tier is rate-limited.

It is a keyword router, not a model. It cannot generalise, and pointing evals
at it would measure nothing - the eval suite uses `FakeLLM` with explicit
scripts for that reason.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from libs.guardrails.normalize import normalize_claim_id, normalize_policy_number

BANNER = "(demo mode - no LLM configured) "

COVERAGE_WORDS = (
    "cover", "covered", "coverage", "deductible", "policy", "excluded",
    "exclusion", "flood", "leak", "burst", "pipe", "jewelry", "jewellery",
    "ring", "laptop", "electronics", "furniture", "appraisal", "limit",
    "water damage", "claim for",
)
FILE_WORDS = ("file a", "submit a", "make a claim", "start a claim", "open a claim")
STATUS_WORDS = ("status", "approved", "decided", "progress", "gone through", "check claim")

_AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")

CLAIM_TYPES = {
    "water": "Water Damage", "pipe": "Water Damage", "flood": "Water Damage",
    "leak": "Water Damage",
    "laptop": "Personal Property", "jewel": "Personal Property",
    "ring": "Personal Property", "electronics": "Personal Property",
    "furniture": "Personal Property", "stolen": "Personal Property",
    "theft": "Personal Property",
}


def _last_human(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if m.type == "human":
            return str(m.content)
    return ""


def _tool_results(messages: Sequence[BaseMessage]) -> list[str]:
    """Tool output from *this* turn only.

    `messages` accumulates across the whole conversation, so scanning it whole
    finds tool results from previous turns and answers with those instead of
    acting on the current question. Everything before the last human message
    belongs to a turn that is already finished.
    """
    turn: list[str] = []
    for m in reversed(messages):
        if m.type == "human":
            break
        if m.type == "tool":
            turn.append(str(m.content))
    return list(reversed(turn))


class ScriptedProvider(BaseChatModel):
    """Routes on keywords, then answers from whatever the tools returned."""

    bound: list[str] = []
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self, tools: Sequence[dict[str, Any] | type | BaseTool], **kwargs: Any
    ) -> Runnable:
        self.bound = [t.name for t in tools if isinstance(t, BaseTool)]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = _last_human(messages)
        lowered = text.lower()
        results = _tool_results(messages)

        # Second pass: tools have run, so answer from their output.
        if results:
            return self._answer(results)

        claim_id = normalize_claim_id(text)
        policy = normalize_policy_number(text)

        if any(w in lowered for w in FILE_WORDS) or (
            "claim" in lowered and policy and _AMOUNT_RE.search(text)
        ):
            return self._call(
                "submit_claim",
                {
                    "policy_number": policy or "POL-1092",
                    "claim_type": next(
                        (v for k, v in CLAIM_TYPES.items() if k in lowered), "Water Damage"
                    ),
                    "amount": (
                        _AMOUNT_RE.search(text).group(1).replace(",", "")
                        if _AMOUNT_RE.search(text)
                        else "0.00"
                    ),
                    "description": text[:500] or "Reported by the policyholder.",
                },
            )

        if claim_id or any(w in lowered for w in STATUS_WORDS):
            return self._call("get_claim_status", {"claim_id": claim_id or "CLM-8821"})

        if any(w in lowered for w in COVERAGE_WORDS):
            return self._call("search_policy_documents", {"query": text, "top_k": 3})

        return self._text(
            BANNER + "I can help with policy coverage, the status of an existing "
            "claim, or filing a new one."
        )

    # ------------------------------------------------------------- helpers

    def _answer(self, results: list[str]) -> ChatResult:
        """Compose a reply from tool output.

        Deliberately extractive: it quotes what the tools returned rather than
        paraphrasing, so a wrong answer here means the tools or retrieval are
        wrong, never that the stand-in invented something.
        """
        import json

        parts: list[str] = []
        for raw in results:
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if "citations" in data:
                for chunk in data.get("chunks", []):
                    body = chunk["text"].split("\n\n", 1)[-1]
                    parts.append(f"{body} ({chunk['citation']})")
            elif data.get("found") is True:
                parts.append(
                    f"Claim {data['claim_id']} on policy {data['policy_number']} "
                    f"is {data['status']} for ${data['amount']:,.2f}."
                )
            elif data.get("found") is False:
                suggestions = ", ".join(data.get("did_you_mean", []))
                parts.append(
                    f"I could not find claim {data['claim_id']}."
                    + (f" Did you mean {suggestions}?" if suggestions else "")
                )
            elif "confirmation_id" in data:
                parts.append(
                    f"Filed. Your confirmation ID is {data['confirmation_id']} "
                    f"({data['readback']}), status {data['status']}."
                )
            elif "error" in data:
                parts.append(f"That did not work: {data.get('detail', data['error'])}")

        return self._text(BANNER + (" ".join(parts) or "I could not find anything on that."))

    def _call(self, name: str, args: dict[str, Any]) -> ChatResult:
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}",
                 "type": "tool_call"}
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _text(self, body: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=body))])
