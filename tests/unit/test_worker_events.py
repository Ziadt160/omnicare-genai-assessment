"""The worker's event emission.

Every test here pins a bug that reached the running system. The worker was the
least-tested component precisely because it needs a queue and a graph, which is
exactly why the bugs collected here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agent.app.settings import AgentSettings
from agent.app.worker import AgentWorker, seed_claims
from libs.contracts import RunEvent, ToolCall


class RecordingQueue:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def publish(self, event: RunEvent) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def of(self, kind: str) -> list[RunEvent]:
        return [e for e in self.events if e.type == kind]


class StubGraph:
    """Returns a fixed final state, like a graph that has already run."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def aget_state(self, config):  # noqa: ANN001
        raise RuntimeError("no snapshot")

    async def ainvoke(self, payload, config):  # noqa: ANN001
        return self.result


def worker_for(result: dict[str, Any]) -> tuple[AgentWorker, RecordingQueue]:
    queue = RecordingQueue()
    settings = AgentSettings(llm_provider="fake", redis_url="", database_url="")
    return AgentWorker(settings, queue, StubGraph(result)), queue


# A plain dict, matching what the graph now stores in state - see the note on
# AgentState about checkpoint serialization.
SEARCH_CALL = ToolCall(
    name="search_policy_documents",
    arguments={"query": "burst pipe"},
    result={"citations": ["sample_policy.md § Section 1: Home Water Damage Coverage"]},
    status="ok",
).model_dump(mode="json")
CITATION = "sample_policy.md § Section 1: Home Water Damage Coverage"


class FakeInterrupt:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value


# --------------------------------------------------------------- happy path

async def test_a_plain_answer_emits_the_full_event_sequence() -> None:
    worker, queue = worker_for({
        "messages": [AIMessage(content="Covered up to $25,000.")],
        "tool_invocations": [SEARCH_CALL],
        "sources": [CITATION],
    })
    await worker.handle({"run_id": "r1", "user_id": "u", "message": "covered?"})

    assert queue.types() == ["started", "tool_end", "token", "sources", "done"]
    assert queue.of("sources")[0].payload["sources"] == [CITATION]
    assert queue.of("token")[0].payload["text"] == "Covered up to $25,000."


async def test_sequence_numbers_are_strictly_increasing() -> None:
    """The client orders by seq; a repeat would render two events as one."""
    worker, queue = worker_for({
        "messages": [AIMessage(content="ok")],
        "tool_invocations": [SEARCH_CALL, SEARCH_CALL],
        "sources": [],
    })
    await worker.handle({"run_id": "r", "user_id": "u", "message": "m"})

    seqs = [e.seq for e in queue.events]
    assert seqs == sorted(set(seqs)), f"duplicate or out-of-order seq: {seqs}"


# ------------------------------------------------------------- confirmation

async def test_tool_calls_before_a_confirmation_are_not_dropped() -> None:
    """Regression.

    A turn can search the policy *and* then propose a write - "is a burst pipe
    covered, and file a $1,200 claim on POL-1092". The interrupt branch used to
    return early, so the search and its citation vanished from the response and
    the answer looked ungrounded.
    """
    worker, queue = worker_for({
        "messages": [AIMessage(content="")],
        "tool_invocations": [SEARCH_CALL],
        "sources": [CITATION],
        "__interrupt__": (
            FakeInterrupt({
                "type": "confirm_write",
                "args": {"policy_number": "POL-1092", "claim_type": "Water Damage"},
                "readback": "Shall I go ahead?",
            }),
        ),
    })
    await worker.handle({"run_id": "r2", "user_id": "u", "message": "file it"})

    assert "tool_end" in queue.types(), "the search before the confirmation was dropped"
    assert queue.of("tool_end")[0].payload["name"] == "search_policy_documents"
    assert queue.of("sources")[0].payload["sources"] == [CITATION]
    assert queue.types()[-1] == "done"


async def test_confirm_event_names_the_tool_as_a_string() -> None:
    """Regression: `tool=value.get("args", {}) and "submit_claim"` evaluated to
    the empty dict when args was empty, so the gateway built a ToolCall named
    "{}"."""
    worker, queue = worker_for({
        "messages": [AIMessage(content="")],
        "tool_invocations": [],
        "sources": [],
        "__interrupt__": (FakeInterrupt({"args": {}, "readback": "Go ahead?"}),),
    })
    await worker.handle({"run_id": "r3", "user_id": "u", "message": "file it"})

    payload = queue.of("confirm")[0].payload
    assert payload["tool"] == "submit_claim"
    assert isinstance(payload["tool"], str)


# ------------------------------------------------------------------ budgets

async def test_an_expired_job_is_dropped_without_running_the_graph() -> None:
    """Spending provider quota on an answer nobody is waiting for is pure waste."""
    worker, queue = worker_for({"messages": [], "tool_invocations": [], "sources": []})
    await worker.handle({
        "run_id": "r4", "user_id": "u", "message": "m", "deadline_at": 0.0,
    })

    assert queue.types() == ["error"]
    assert "expired" in queue.of("error")[0].payload["message"].lower()


# -------------------------------------------------------------------- seed

def test_seed_copies_the_fixture_onto_an_empty_volume(tmp_path) -> None:
    """Regression: a named volume starts empty, and nothing copied the fixture
    in - so every claim lookup returned "not found" in the running stack."""
    seed = tmp_path / "seed.json"
    seed.write_text('[{"claim_id": "CLM-8821", "policy_number": "POL-1092", '
                    '"claim_type": "Water Damage", "status": "Approved", '
                    '"amount": 3500.0, "description": ""}]', encoding="utf-8")
    target = tmp_path / "vol" / "mock_claims.json"

    seed_claims(AgentSettings(claims_path=str(target), claims_seed_path=str(seed)))
    assert "CLM-8821" in target.read_text(encoding="utf-8")


def test_seed_never_overwrites_filed_claims(tmp_path) -> None:
    """Once a policyholder has filed a claim that file is the system of record.
    A restart must not silently discard it."""
    seed = tmp_path / "seed.json"
    seed.write_text("[]", encoding="utf-8")
    target = tmp_path / "mock_claims.json"
    target.write_text('[{"claim_id": "CLM-9999"}]', encoding="utf-8")

    seed_claims(AgentSettings(claims_path=str(target), claims_seed_path=str(seed)))
    assert "CLM-9999" in target.read_text(encoding="utf-8")


def test_missing_seed_yields_an_empty_store_not_a_crash(tmp_path) -> None:
    target = tmp_path / "mock_claims.json"
    seed_claims(AgentSettings(
        claims_path=str(target), claims_seed_path=str(tmp_path / "absent.json")
    ))
    assert target.read_text(encoding="utf-8") == "[]"
