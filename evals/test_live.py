"""The live eval suite. Marked, and excluded from CI.

    docker compose up -d
    pytest evals/test_live.py -m live -q -s

Runs the same 29 cases as the deterministic gate, but against the **running
stack over HTTP** with whatever model is configured. That is the difference
that matters: `test_gates.py` scripts the model's decisions and therefore
measures the *graph* - guard, routing, grounding, confirmation. This measures
whether the tool docstrings and the system prompt actually steer a real model
into making those decisions in the first place.

It is not a CI gate, for two reasons that are both about honesty rather than
convenience: a free-tier quota is not a stable dependency, and a local 7B model
takes 15-40 s per turn, so the whole suite is minutes rather than seconds. The
numbers belong in the README beside the deterministic ones - published
together, they say more than either alone.
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

from evals.runner import CaseResult, EvalCase, Outcome, check, format_report, load_cases

pytestmark = pytest.mark.live

BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8080")
# A local 7B on CPU is tens of seconds per call and a ReAct turn is two.
TIMEOUT = float(os.environ.get("LIVE_TIMEOUT_S", "300"))

CASES = load_cases()

# Unique per run. Conversations are keyed on user_id and the checkpointer
# resumes the most recent one, so a fixed id makes the second run inherit the
# first run's history - and any confirmation left paused by it. The symptom is
# baffling: every tool-calling case reports "tools not called", because the
# graph is resuming an interrupt rather than starting a turn.
RUN = uuid.uuid4().hex[:8]


def _reachable() -> bool:
    try:
        return httpx.get(f"{BASE}/api/v1/health", timeout=3).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _reachable(), reason=f"no stack at {BASE}"),
]


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE, timeout=TIMEOUT) as c:
        yield c


def _ask(client: httpx.Client, user: str, message: str, channel: str, confidence) -> dict:
    """One turn. An error response becomes a recorded failure, not an abort.

    A 502 or 429 on one case must not lose the other twenty-eight - the whole
    point of the run is the scorecard, and a provider hiccup mid-suite is
    exactly the condition it exists to survive.
    """
    payload = {"user_id": user, "message": message, "channel": channel}
    if confidence is not None:
        payload["stt_confidence"] = confidence

    r = client.post("/api/v1/chat", json=payload)
    if r.status_code != 200:
        detail = ""
        try:
            detail = str(r.json().get("detail", ""))[:160]
        except Exception:
            detail = r.text[:160]
        return {
            "response": f"[HTTP {r.status_code}] {detail}",
            "sources": [],
            "tool_calls": [],
            "_http_error": r.status_code,
        }
    return r.json()


def run_live(client: httpx.Client, case: EvalCase) -> tuple[Outcome, float]:
    """One case against the deployed system.

    A fresh user id per case: the checkpointer keys threads on the
    conversation, so a shared id would let a paused confirmation from one case
    resume inside the next.
    """
    user = f"live{RUN}{case.id.replace('-', '').lower()}"
    started = time.perf_counter()

    body = _ask(client, user, case.message, case.channel, case.stt_confidence)
    awaiting = any(
        t.get("status") == "awaiting_confirmation" for t in body.get("tool_calls", [])
    )

    if case.followup:
        body = _ask(client, user, case.followup, case.channel, None)
        awaiting = any(
            t.get("status") == "awaiting_confirmation" for t in body.get("tool_calls", [])
        )

    calls = body.get("tool_calls", [])
    outcome = Outcome(
        text=body.get("response", ""),
        sources=body.get("sources", []),
        tools_called=[t["name"] for t in calls],
        tool_args={t["name"]: t.get("arguments", {}) for t in calls},
        # The gateway does not expose the guard verdict, so a refusal is
        # recognised by its shape: the fixed refusal text, no tools, no sources.
        blocked=(
            "can't help with that request" in body.get("response", "").lower()
            and not calls
        ),
        awaiting_confirmation=awaiting,
        # A paused turn returns its readback as the response body - the same
        # string the confirm panel renders. See frontend/src/app.js.
        confirmation_readback=body.get("response", "") if awaiting else "",
        claim_written=any(
            t["name"] == "submit_claim" and t.get("status") == "ok" for t in calls
        ),
        transport_error=body.get("_http_error"),
    )
    return outcome, time.perf_counter() - started


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_live_case(client, case: EvalCase, record_property) -> None:
    """Reported, not gated.

    A real model will fail some of these, and that is information rather than a
    build failure. Failures print with the model's actual answer so the fix is
    obvious - almost always a tool docstring rather than graph code.
    """
    outcome, elapsed = run_live(client, case)
    record_property("elapsed_s", round(elapsed, 1))

    if outcome.transport_error is not None:
        pytest.skip(
            f"{case.id}: HTTP {outcome.transport_error} - never reached the model. "
            f"{outcome.text[:120]}"
        )

    failures = [f for f in check(case, outcome) if "confirmation tier" not in f]
    if failures:
        pytest.xfail(
            f"{case.id} [{case.bucket}] {elapsed:.1f}s\n"
            f"  message : {case.message}\n"
            f"  answer  : {outcome.text[:240]}\n"
            f"  tools   : {outcome.tools_called}\n"
            f"  sources : {outcome.sources}\n"
            f"  failed  : " + "; ".join(failures)
        )


def test_live_report(client, capsys) -> None:
    """Run every case once and print the scorecard for the README.

    Never asserts. The deterministic suite is the gate; this is the measurement
    that goes next to it.
    """
    results: list[CaseResult] = []
    total = 0.0

    for case in CASES:
        outcome, elapsed = run_live(client, case)
        total += elapsed
        # `confirmation_tier` is graph-internal and deliberately not exposed on
        # the API, so it cannot be observed over HTTP. The deterministic suite
        # asserts it directly; the readback it drives *is* observable and is
        # still checked here.
        failures = (
            []
            if outcome.transport_error is not None
            else [f for f in check(case, outcome) if "confirmation tier" not in f]
        )
        results.append(CaseResult(case, outcome, failures))

    provider = os.environ.get("LLM_PROVIDER", "?")
    model = os.environ.get("LLM_MODEL", "?")

    with capsys.disabled():
        print(f"\n\nLIVE RUN  provider={provider}  model={model}")
        print(f"{len(CASES)} cases in {total:.0f}s  "
              f"({total / max(len(CASES), 1):.1f}s per case)")
        print(format_report(results))
