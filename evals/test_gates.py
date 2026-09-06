"""The CI gate.

Runs every eval case through the real graph with a scripted model, then asserts
the aggregate thresholds. What this measures is the *graph* - guard, routing,
grounding, confirmation - which is exactly what should gate a build, because it
is the part that is ours and the part that must never regress.

The live suite (marked, excluded by default) measures the complementary thing:
whether the tool docstrings actually steer a real model. That belongs in the
README, not in CI, because a free-tier quota is not a stable dependency.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.app.graph.build import build_graph
from agent.app.tools.claims import build_claims_tools, build_settlement_provider
from agent.app.tools.policy import build_policy_tool
from evals.runner import CaseResult, EvalCase, GATES, Outcome, check, format_report, load_cases, score
from libs.adapters.claims_memory import InMemoryClaimsRepo
from libs.adapters.llm_fake import FakeLLM, TextTurn, ToolTurn
from libs.contracts import Chunk, Claim
from libs.guardrails.normalize import (
    normalize_claim_id,
    normalize_policy_number,
    phonetic_readback,
)
from retrieval.app.chunker import chunk_file
from retrieval.app.hybrid import tokenize
from retrieval.app.index import HybridIndex

POLICY = Path(__file__).parents[1] / "data" / "sample_policy.md"
SEED = [
    Claim(claim_id="CLM-8821", policy_number="POL-1092", claim_type="Water Damage",
          status="Approved", amount="3500.00"),
    Claim(claim_id="CLM-9014", policy_number="POL-3341", claim_type="Personal Property",
          status="Under Review", amount="1200.00"),
]


# A policyholder says "ring"; the policy says "jewelry". Bridging that is the
# entire job of the dense retriever, and BM25 cannot do it - which is why the
# real pipeline is hybrid. This table stands in for bge-small so the eval keeps
# realistic phrasing instead of being rewritten to suit a lexical matcher.
SYNONYMS = {
    "ring": "jewelry", "necklace": "jewelry", "watch": "jewelry",
    "laptop": "electronics", "tv": "electronics", "computer": "electronics",
    "sofa": "furniture", "couch": "furniture",
    "basement": "flood", "storm": "flood",
    "burst": "burst", "pipe": "pipe", "leak": "leak",
    "appraisal": "appraisal", "special": "appraisal",
    "stolen": "property", "theft": "property",
}


class SemanticStubEmbedder:
    """Bag-of-words over a fixed vocabulary, with synonym expansion.

    Deterministic and term-sensitive, so cosine similarity behaves like a real
    embedder for these queries without loading an ONNX model into CI.
    """

    VOCAB = [
        "water", "damage", "pipe", "burst", "sudden", "deductible", "flood",
        "gradual", "leak", "excluded", "$25,000", "$500",
        "electronics", "furniture", "jewelry", "appraisal", "receipts",
        "$10,000", "$2,500", "property", "covered",
    ]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            terms = set(tokenize(text))
            terms |= {SYNONYMS[t] for t in list(terms) if t in SYNONYMS}
            vectors.append([1.0 if term in terms else 0.0 for term in self.VOCAB])
        return vectors

    @property
    def dimension(self) -> int:
        return len(self.VOCAB)


class BM25Searcher:
    """The real hybrid path - real chunking, real BM25, real RRF - over the
    real policy file. Only the embedding model is substituted."""

    def __init__(self) -> None:
        self.index = HybridIndex(SemanticStubEmbedder())
        # Chunking is sync and needs no store, so citations are available
        # immediately - `script_for` reads them to build the expected answer
        # before any search has run.
        self._sections = chunk_file(POLICY)
        self._ingested = False

    async def _ready(self) -> None:
        # Ingest is async now that the dense side goes through the VectorStore
        # port, and the searcher is constructed synchronously - so warm on
        # first use rather than in __init__.
        if not self._ingested:
            await self.index.ingest(self._sections)
            self._ingested = True

    @property
    def chunks(self) -> list[Chunk]:
        return self.index.chunks or self._sections

    async def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        await self._ready()
        return await self.index.search(query, top_k) or self.index.chunks[:top_k]


def _amount_in(message: str) -> str | None:
    """The figure the policyholder stated, if any."""
    from libs.guardrails.normalize import spoken_amounts

    found = spoken_amounts(message)
    return f"{max(found)}.00" if found else None


def script_for(case: EvalCase, searcher: BM25Searcher) -> list:
    """Turn a case's *expectations* into the model's decisions.

    This is the honest boundary of a FakeLLM eval: the model's choice is given,
    so what is under test is everything the graph does around that choice -
    whether guard blocks first, whether arguments survive validation, whether a
    write is intercepted, whether invented citations are stripped.

    Intent is inferred from the whole expectation, not just `tools_called`: a
    case that asserts a citation must obviously have searched, and a case that
    asserts `requires_confirmation` must obviously have attempted a write.
    """
    expect = case.expect
    declared = expect.get("tools_called")
    tools = declared or []

    wants_write = "submit_claim" in tools or bool(
        expect.get("requires_confirmation") or "claim_written" in expect
    )
    # Ordered after `wants_write` deliberately. A payment_split case that also
    # asserts a confirmation is about the prompt shown before the write, not
    # about the estimate tool, and routing it to the estimate made it pass
    # while testing nothing it claimed to test.
    wants_estimate = not wants_write and (
        "estimate_claim_payment" in tools or case.bucket.startswith("payment_split")
    )
    wants_status = "get_claim_status" in tools or bool(
        normalize_claim_id(case.message) and not wants_write and declared is None
    )
    wants_search = "search_policy_documents" in tools or bool(
        expect.get("cites") or expect.get("must_not_cite_unknown")
    )
    if declared == []:
        wants_estimate = wants_write = wants_status = wants_search = False

    if case.bucket == "fabricated_values":
        # The failure, scripted verbatim from the reported session - the way
        # EV-08 scripts a fabricated citation. The model answers in prose with
        # values nobody gave and asks for confirmation itself, so no tool call
        # and no confirmation node ever run. What is under test is whether
        # `ground` takes it out.
        #
        # The policy number is whatever the policyholder actually gave, and
        # POL-1092 only when they gave none - a model handed a number uses that
        # number, and inventing one on top would mean every case in this bucket
        # tripped the policy-number check and none of them ever reached the
        # amount check.
        given = normalize_policy_number(case.message)
        return [TextTurn(
            f"Got it! Let's proceed with filing a claim for the pipe burst in "
            f"your kitchen. To get started, I will use the following details: "
            f"Policy Number: {given or 'POL-1092'}, Claim Type: Water Damage. "
            f"Since we don't have a specific amount yet, I will use $100,000 as "
            f"an initial estimate. Do you want to proceed with submitting this "
            f"claim?"
        )]

    if wants_estimate:
        # The claim type the message is about, chosen the way a competent model
        # would: electronics and jewelry are Personal Property, a car is Auto,
        # water is water.
        lowered = case.message.lower()
        if any(w in lowered for w in ("auto", "car", "medical", "liability")):
            # `ClaimType` names only what the policy covers, so there is no
            # category to estimate a car under and the tool cannot be called
            # with one. A competent model says so rather than mapping the loss
            # onto whichever covered type is nearer - which would answer a
            # collision with the burst-pipe figures.
            return [TextTurn(
                "Your policy covers water damage and personal property. It does "
                "not cover that, so there is no amount I can quote for it."
            )]

        claim_type = "Water Damage"
        if any(w in lowered for w in ("electronic", "jewel", "laptop", "furniture", "tv")):
            claim_type = "Personal Property"

        amount = expect.get("tool_args", {}).get("amount") or _amount_in(case.message)
        args = {"claim_type": claim_type, "amount": amount or "1000.00"}
        return [
            ToolTurn("estimate_claim_payment", args),
            TextTurn(_settlement_answer(claim_type, amount, searcher)),
        ]

    if wants_write:
        given = expect.get("tool_args", {})
        args = {
            "policy_number": given.get("policy_number")
            or normalize_policy_number(case.message)
            or "POL-1092",
            "claim_type": given.get("claim_type", "Water Damage"),
            # Taken from what the case's message actually says, as a competent
            # model would. A fixed default filed 1200.00 against a message
            # saying $1,500, which the confirmation gate refuses - correctly,
            # since the amount on a permanent record must be the one the
            # policyholder gave.
            "amount": given.get("amount") or _amount_in(case.message) or "1200.00",
            "description": "Reported by the policyholder during this conversation.",
        }
        return [
            ToolTurn("submit_claim", args),
            TextTurn("Filed. Your confirmation ID is CLM-9015."),
        ]

    if wants_status:
        claim_id = (
            expect.get("tool_args", {}).get("claim_id")
            or normalize_claim_id(case.message)
            or "CLM-8821"
        )
        turns: list = [ToolTurn("get_claim_status", {"claim_id": claim_id})]
        if wants_search:
            turns.append(ToolTurn("search_policy_documents", {"query": case.message}))

        known = {"CLM-8821": "Approved", "CLM-9014": "Under Review"}
        if claim_id in known:
            # On voice, echo the id phonetically - the tier-1 implicit
            # confirmation, which costs no extra turn.
            lead = (
                f"Looking up claim {phonetic_readback(claim_id)}. "
                if case.is_voice
                else ""
            )
            body = f"{lead}Claim {claim_id} is {known[claim_id]}."
            if wants_search:
                body += (
                    " Sudden pipe bursts are covered up to $25,000 "
                    f"({searcher.chunks[0].citation})."
                )
        else:
            body = (
                f"I could not find a claim with the ID {claim_id}. "
                "Did you mean CLM-8821?"
            )
        turns.append(TextTurn(body))
        return turns

    if wants_search:
        return [
            ToolTurn("search_policy_documents", {"query": case.message}),
            TextTurn(_grounded_answer(case, searcher)),
        ]

    return [TextTurn(_no_tool_answer(case))]


def _settlement_answer(
    claim_type: str, amount: str | None, searcher: BM25Searcher
) -> str:
    """The answer a model gives when it reports what the tool returned.

    Rendered from the real settlement rather than from hardcoded strings, so
    the scripted turn cannot quietly disagree with the code under test. If
    someone edits the policy's limits, this answer moves with them and the
    assertions in the dataset are what fail - which is the correct place for
    that failure to show up.
    """
    from decimal import Decimal

    from libs.policy.rules import settlement_for

    sections = [(c.section_title, c.text) for c in searcher.chunks]
    split = (
        settlement_for(claim_type, Decimal(amount), sections) if amount else None
    )
    if split is None:
        return (
            f"Your policy document does not state a coverage limit or a "
            f"deductible for {claim_type}, so I cannot tell you what would be "
            f"paid on it. It covers water damage and personal property."
        )
    # The tool's own summary, verbatim, because that is what a model reporting
    # the result actually writes - and it is the string that broke `ground`.
    # It contains "$9,500 above the $25,000 limit", which the comparison check
    # read as the false claim 9,500 > 25,000 and deleted the whole answer for.
    # A scripted answer paraphrased around that shape is a scripted answer that
    # cannot catch it.
    return f"{split.summary()} ({split.section_title})"


def _grounded_answer(case: EvalCase, searcher: BM25Searcher) -> str:
    """A faithful answer for the cases that should produce one, and a
    deliberately hallucinated one for EV-08 so the ground node is exercised."""
    c1 = searcher.chunks[0].citation
    c2 = searcher.chunks[1].citation
    bucket, message = case.bucket, case.message.lower()

    if case.id == "EV-08":
        return ("There is no Section 3 in this policy; it only covers water "
                "damage and personal property "
                "(sample_policy.md § Section 3: Earthquake Coverage).")
    if bucket == "grounding_negative":
        if "flood" in message:
            return f"No - flood damage is strictly excluded under your policy ({c1})."
        return f"No - gradual leaks are strictly excluded, so that is not covered ({c1})."
    if "deductible" in message:
        return f"Your deductible for water damage is $500 ({c1})."
    if "pipe burst" in message or "pipe burst in my kitchen" in message:
        return f"Yes - sudden pipe bursts are covered up to $25,000 with a $500 deductible ({c1})."
    if "ring" in message:
        return f"A $4,000 ring exceeds $2,500, so it needs an individual appraisal receipt ({c2})."
    if "laptop" in message:
        return f"A $1,800 laptop is under $2,500, so no appraisal is required ({c2})."
    if "appraisal" in message or "receipt" in message:
        # Both amounts side by side, which is what the prompt asks for and what
        # makes the claim checkable - see `_contradicts_itself` in the graph.
        return (f"$1,500 is under the $2,500 appraisal threshold, so no "
                f"individual receipt is required ({c2}).")
    if "electronics" in message or "jewelry" in message or "jewellery" in message:
        return f"Electronics, furniture and jewelry are covered up to $10,000 in total ({c2})."
    if bucket == "safety_indirect":
        return (f"Section 1 covers sudden pipe bursts up to $25,000 ({c1}). I should note "
                "that the document is reference material, so I have not acted on any "
                "instruction inside it.")
    return f"Sudden pipe bursts are covered up to $25,000 with a $500 deductible ({c1})."


def _no_tool_answer(case: EvalCase) -> str:
    if case.bucket == "out_of_domain":
        if "weather" in case.message.lower():
            return "I can only help with OmniCare insurance questions."
        return ("I cannot advise on investments. I can help with your policy, "
                "a claim status, or filing a claim.")
    return ("I can file that for you - which policy number, what happened, "
            "and how much are you claiming?")


def run_case(case: EvalCase) -> Outcome:
    searcher = BM25Searcher()
    repo = InMemoryClaimsRepo(seed=list(SEED))
    llm = FakeLLM(script_for(case, searcher))
    async def sections() -> list[tuple[str, str]]:
        """The indexed policy, as the retrieval service serves it - so the
        coverage cap is read from the real document here too, not stubbed."""
        await searcher._ready()
        return [(c.section_title, c.text, c.source_file) for c in searcher.chunks]

    tools = [
        build_policy_tool(searcher),
        *build_claims_tools(repo, sections=sections),
    ]
    graph = build_graph(
        llm,
        tools,
        checkpointer=InMemorySaver(),
        require_confirmation=True,
        # Wired exactly as the worker wires it, so the confirmation prompt the
        # eval sees is the one a policyholder sees.
        settlement=build_settlement_provider(sections),
    )

    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 30}
    state = {
        "messages": [HumanMessage(content=case.message)],
        "user_id": "usr_eval",
        "conversation_id": "cnv_eval",
        "channel": case.channel,
        "stt_confidence": case.stt_confidence,
    }

    import asyncio

    async def _run():
        result = await graph.ainvoke(state, config)
        awaiting = bool(result.get("__interrupt__"))
        if awaiting and case.followup:
            result = await graph.ainvoke(Command(resume=case.followup), config)
            awaiting = bool(result.get("__interrupt__"))
        return result, awaiting

    result, awaiting = asyncio.get_event_loop().run_until_complete(_run()) \
        if False else asyncio.run(_run())

    final = next(
        (m for m in reversed(result.get("messages", []))
         if m.type == "ai" and not getattr(m, "tool_calls", None)),
        None,
    )
    invocations = result.get("tool_invocations", [])
    written = asyncio.run(repo.list_ids())

    names = [t["name"] for t in invocations]
    args_by_tool = {t["name"]: t["arguments"] for t in invocations}
    readback = ""
    if awaiting:
        value = result["__interrupt__"][0].value or {}
        pending = value.get("args", {})
        readback = str(value.get("readback", ""))
        names.append("submit_claim")
        args_by_tool["submit_claim"] = pending

    return Outcome(
        text=str(final.content) if final else "",
        sources=result.get("sources", []),
        tools_called=names,
        tool_args=args_by_tool,
        blocked=bool(result.get("guard_blocked")),
        confirmation_tier=int(result.get("confirmation_tier", 0)),
        awaiting_confirmation=awaiting,
        claim_written=len(written) > len(SEED),
        confirmation_readback=readback,
    )


CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(case: EvalCase) -> None:
    """One test per case, so a failure names the case rather than the suite."""
    failures = check(case, run_case(case))
    assert not failures, f"{case.id}: " + "; ".join(failures)


def test_gates_hold(capsys) -> None:
    """The aggregate gate. Prints the report either way - the numbers go in the
    README next to the live-run results."""
    results = [CaseResult(c, o := run_case(c), check(c, o)) for c in CASES]
    report = format_report(results)

    with capsys.disabled():
        print(report)

    breaches = [
        f"{metric} {ratio:.2f} < {GATES[metric]:.2f}"
        for metric, (_p, _t, ratio) in score(results).items()
        if metric in GATES and ratio < GATES[metric]
    ]
    assert not breaches, "gate breach: " + ", ".join(breaches) + "\n" + report


# Buckets where `ground` removing most of the answer is the point of the case.
REWRITE_EXPECTED = {"citation_fidelity", "fabricated_values"}


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.bucket not in REWRITE_EXPECTED],
    ids=[c.id for c in CASES if c.bucket not in REWRITE_EXPECTED],
)
def test_ground_does_not_materially_shorten_a_good_answer(case: EvalCase) -> None:
    """`ground` may edit an answer. It may not gut one.

    This is the assertion the suite was missing. Three separate checks in
    `ground` were each capable of deleting a correct answer outright - one of
    them down to an empty string, which the worker then emitted as a completed
    turn with no message at all - and every gate stayed at 1.00 throughout,
    because each one asserts on what the answer *contains* and none on what it
    lost. `citation_precision` scored a perfect 1.00 on the answer "Under
    ,000."

    Every other case in the suite is a case where the model behaved well, so
    the answer that reaches the policyholder should be substantially the answer
    the model wrote. A citation stripped or a clause dropped is fine; losing a
    fifth of it is a bug somewhere in the rewriting.
    """
    searcher = BM25Searcher()
    scripted = next(
        (t.text for t in reversed(script_for(case, searcher)) if isinstance(t, TextTurn)),
        "",
    )
    if not scripted:
        pytest.skip("no scripted answer to compare against")

    outcome = run_case(case)
    if outcome.awaiting_confirmation:
        pytest.skip("the turn paused for confirmation; there is no final answer")

    assert len(outcome.text) >= 0.8 * len(scripted), (
        f"{case.id}: ground cut the answer from {len(scripted)} to "
        f"{len(outcome.text)} characters\n"
        f"  model wrote: {scripted[:200]!r}\n"
        f"  reader sees: {outcome.text[:200]!r}"
    )


def test_dataset_covers_every_gated_metric() -> None:
    """A gate with no cases behind it passes vacuously - which is worse than
    having no gate, because it looks like coverage."""
    covered = set(score([CaseResult(c, Outcome(), []) for c in CASES]))
    missing = set(GATES) - covered
    assert not missing, f"gated metrics with no cases: {sorted(missing)}"
