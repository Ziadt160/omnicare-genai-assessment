"""The graded contract.

`POST /api/v1/chat` and `GET /api/v1/health` are specified in the brief and
their shapes are fixed. These tests exist to catch the failure mode where a
later feature quietly changes one of them - an added required field, a renamed
key, a health body that gained an extra property.

A stub worker drains the job queue and emits the real event vocabulary, so the
full enqueue-then-await-the-result-stream path runs end to end without Redis or
an agent container. Everything shares one event loop via ASGITransport.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import httpx
import pytest
import pytest_asyncio

from gateway.app.deps import Deps
from gateway.app.main import app
from gateway.app.settings import GatewaySettings
from libs.adapters.conversations_memory import InMemoryConversationRepo
from libs.adapters.queue_memory import InMemoryQueue, InMemoryRateLimiter
from libs.contracts import RunEvent

CITATION = "sample_policy.md § Section 1: Home Water Damage Coverage"
COVERAGE_Q = "A pipe burst in my kitchen. Am I covered?"
FILE_CLAIM_Q = "File a water damage claim on POL-1092 for $1,200"

SCRIPT: dict[str, list[RunEvent]] = {
    COVERAGE_Q: [
        RunEvent(run_id="", type="started", seq=0),
        RunEvent(run_id="", type="tool_end", seq=1, payload={
            "name": "search_policy_documents",
            "arguments": {"query": "burst pipe coverage"},
            "result": {"citations": [CITATION]},
            "status": "ok",
        }),
        RunEvent(run_id="", type="token", seq=2, payload={
            "text": "Sudden pipe bursts are covered up to $25,000 with a $500 "
                    f"deductible ({CITATION})."
        }),
        RunEvent(run_id="", type="sources", seq=3, payload={"sources": [CITATION]}),
        RunEvent(run_id="", type="done", seq=4, payload={"trace_id": "trace-abc"}),
    ],
    FILE_CLAIM_Q: [
        RunEvent(run_id="", type="confirm", seq=0, payload={
            "tool": "submit_claim",
            "args": {"policy_number": "POL-1092", "claim_type": "Water Damage",
                     "amount": "1200.00", "description": "Pipe burst in the kitchen."},
            "readback": "I'm about to file a Water Damage claim on policy "
                        "P-O-L, one zero nine two for $1200.00. Shall I go ahead?",
        }),
        RunEvent(run_id="", type="done", seq=1),
    ],
    "__default__": [
        RunEvent(run_id="", type="token", seq=0, payload={"text": "Hello."}),
        RunEvent(run_id="", type="done", seq=1),
    ],
    "__silent__": [],
}


class StubWorker:
    """Stands in for the agent service: consumes jobs, emits the same events."""

    def __init__(self, queue: InMemoryQueue) -> None:
        self.queue = queue
        self.seen: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            job = await self.queue.jobs.get()
            self.seen.append(job)
            key = job["message"] if job["message"] in SCRIPT else "__default__"
            if job["message"] == "stay silent":
                key = "__silent__"
            for event in SCRIPT[key]:
                await self.queue.publish(event.model_copy(update={"run_id": job["run_id"]}))

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()


@pytest_asyncio.fixture()
async def ctx() -> AsyncIterator[tuple[httpx.AsyncClient, StubWorker]]:
    queue = InMemoryQueue()
    app.state.deps = Deps(
        settings=GatewaySettings(redis_url="", database_url="", run_timeout_s=3.0),
        queue=queue,
        conversations=InMemoryConversationRepo(),
        rate_limiter=InMemoryRateLimiter(),
    )
    worker = StubWorker(queue)
    worker.start()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        yield client, worker
    await worker.stop()


@pytest_asyncio.fixture()
async def client(ctx) -> httpx.AsyncClient:
    return ctx[0]


# ------------------------------------------------------------------ health

async def test_health_returns_exactly_the_specified_body(client) -> None:
    """The brief specifies {"status": "healthy"}. Not a superset."""
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


async def test_deep_health_reports_dependencies_separately(client) -> None:
    r = await client.get("/api/v1/health/deep")
    assert r.status_code == 200
    assert "dependencies" in r.json()


# ------------------------------------------------------------- happy path

async def test_coverage_question_returns_the_graded_shape(client) -> None:
    r = await client.post("/api/v1/chat", json={"user_id": "usr_123", "message": COVERAGE_Q})
    assert r.status_code == 200

    body = r.json()
    assert set(body) == {"response", "sources", "tool_calls", "conversation_id", "trace_id"}
    assert "$25,000" in body["response"]
    assert body["sources"] == [CITATION]
    assert body["tool_calls"][0]["name"] == "search_policy_documents"
    assert r.headers["X-Trace-Id"] == "trace-abc"


async def test_specified_two_field_payload_still_works(client) -> None:
    """The exact payload from the brief, after every additive field."""
    r = await client.post("/api/v1/chat", json={"user_id": "usr_123", "message": "hello"})
    assert r.status_code == 200
    assert r.json()["response"] == "Hello."


async def test_job_reaches_the_worker_with_defaults_applied(ctx) -> None:
    client, worker = ctx
    await client.post("/api/v1/chat", json={"user_id": "usr_123", "message": "hello"})
    job = worker.seen[-1]
    assert job["user_id"] == "usr_123"
    assert job["channel"] == "text"
    assert job["stt_confidence"] is None
    assert job["conversation_id"].startswith("cnv_")


async def test_pending_write_is_surfaced_as_a_tool_call(client) -> None:
    """A claim awaiting confirmation must be visible in tool_calls, not hidden
    in prose - the caller needs to know a write is pending."""
    r = await client.post("/api/v1/chat", json={"user_id": "usr_123", "message": FILE_CLAIM_Q})
    body = r.json()
    call = body["tool_calls"][0]
    assert call["name"] == "submit_claim"
    assert call["status"] == "awaiting_confirmation"
    assert "Shall I go ahead?" in body["response"]


async def test_history_is_recorded_for_both_turns(ctx) -> None:
    client, _ = ctx
    r = await client.post("/api/v1/chat", json={"user_id": "usr_hist", "message": COVERAGE_Q})
    cid = r.json()["conversation_id"]

    messages = (await client.get(f"/api/v1/conversations/{cid}/messages")).json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["sources"] == [CITATION]
    assert messages[1]["trace_id"] == "trace-abc"

    conversations = (await client.get("/api/v1/conversations/usr_hist")).json()
    assert conversations[0]["id"] == cid


async def test_reusing_a_conversation_id_appends(client) -> None:
    first = await client.post("/api/v1/chat", json={"user_id": "usr_x", "message": "hello"})
    cid = first.json()["conversation_id"]
    second = await client.post(
        "/api/v1/chat", json={"user_id": "usr_x", "message": "hello", "conversation_id": cid}
    )
    assert second.json()["conversation_id"] == cid
    messages = (await client.get(f"/api/v1/conversations/{cid}/messages")).json()
    assert len(messages) == 4


# -------------------------------------------------------------- validation

@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"user_id": "usr_123"}, "missing message"),
        ({"message": "hello"}, "missing user_id"),
        ({"user_id": "", "message": "hello"}, "empty user_id"),
        ({"user_id": "usr_123", "message": ""}, "empty message"),
        ({"user_id": "usr 123", "message": "hi"}, "space in user_id"),
        ({"user_id": "usr_123", "message": "hi", "admin": True}, "unknown key"),
        ({"user_id": "usr_123", "message": "hi", "channel": "telepathy"}, "bad channel"),
        ({"user_id": "usr_123", "message": "hi", "stt_confidence": 1.5}, "confidence > 1"),
        ({"user_id": "usr_123", "message": "x" * 4001}, "message too long"),
    ],
)
async def test_invalid_payloads_are_rejected(client, payload, why: str) -> None:
    """`extra="forbid"` makes an unknown key a 422 rather than a silent drop -
    the difference between validation and decoration."""
    r = await client.post("/api/v1/chat", json=payload)
    assert r.status_code == 422, why


# ------------------------------------------------------------- failure mode

async def test_a_silent_agent_returns_504_not_an_empty_200(client) -> None:
    """A blank 200 is the worst possible failure here: the caller cannot tell
    a refusal from a crash."""
    r = await client.post("/api/v1/chat", json={"user_id": "usr_123", "message": "stay silent"})
    assert r.status_code == 504
    assert "did not respond" in r.json()["detail"]


# ------------------------------------------------------------ openapi shape

async def test_openapi_documents_the_graded_contract(client) -> None:
    spec = (await client.get("/openapi.json")).json()
    assert "/api/v1/chat" in spec["paths"]
    assert "/api/v1/health" in spec["paths"]

    response_schema = spec["components"]["schemas"]["ChatResponse"]
    for field in ("response", "sources", "tool_calls"):
        assert field in response_schema["properties"], field

    request_schema = spec["components"]["schemas"]["ChatRequest"]
    assert set(request_schema["required"]) == {"user_id", "message"}, (
        "only the two specified fields may be required; everything additive "
        "must carry a default"
    )


async def test_health_schema_has_no_extra_properties(client) -> None:
    """Guards the literal graded body against a future field creeping in."""
    spec = (await client.get("/openapi.json")).json()
    assert set(spec["components"]["schemas"]["HealthResponse"]["properties"]) == {"status"}


async def test_request_schema_carries_runnable_examples(client) -> None:
    """Examples populate Swagger and the generated Postman collection, so they
    cannot drift from the model."""
    spec = (await client.get("/openapi.json")).json()
    examples = spec["components"]["schemas"]["ChatRequest"].get("examples", [])
    assert examples and examples[0]["user_id"] == "usr_123"
