"""The agent worker.

Consumes `jobs:chat` through a Redis consumer group and publishes events onto
`stream:{run_id}`. Nothing else talks to the graph - the gateway and the voice
worker both enqueue, so REST, WebSocket and voice all exercise the same code
path and the same evals cover all three.

Run N of these and the consumer group distributes the work:

    docker compose up --scale agent=4
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from libs.contracts import RunEvent
from libs.errors import RunTimeout
from libs.observability import otel
from .graph.build import build_graph
from .settings import AgentSettings

log = logging.getLogger("omnicare.agent")
CONSUMER = f"{socket.gethostname()}-{os.getpid()}"


class AgentWorker:
    def __init__(self, settings: AgentSettings, queue: Any, graph: Any) -> None:
        self.settings = settings
        self.queue = queue
        self.graph = graph
        self._running = False

    # ------------------------------------------------------------ events

    async def _emit(self, run_id: str, kind: str, seq: int, **payload: Any) -> None:
        await self.queue.publish(
            RunEvent(run_id=run_id, type=kind, seq=seq, payload=payload)  # type: ignore[arg-type]
        )

    # -------------------------------------------------------------- run

    async def handle(self, job: dict[str, Any]) -> None:
        run_id = job["run_id"]
        seq = 0
        started = time.perf_counter()

        deadline_at = job.get("deadline_at")
        if deadline_at is not None and time.time() > float(deadline_at):
            # Nobody is waiting for this any more. Spending provider quota on
            # an answer no one will read is pure waste.
            await self._emit(run_id, "error", seq, message="Request expired in queue.")
            return

        await self._emit(run_id, "started", seq)
        seq += 1
        trace_id: str | None = None

        config = {
            "configurable": {"thread_id": job.get("conversation_id") or job["user_id"]},
            "recursion_limit": self.settings.max_graph_iterations * 4,
        }

        state = {
            "messages": [HumanMessage(content=job["message"])],
            "user_id": job["user_id"],
            "conversation_id": job.get("conversation_id", ""),
            "channel": job.get("channel", "text"),
            "stt_confidence": job.get("stt_confidence"),
        }

        try:
            snapshot = await self.graph.aget_state(config)
            resuming = bool(getattr(snapshot, "next", ()) )
        except Exception:
            resuming = False

        try:
            # A paused confirmation is resumed by the next message, whatever it
            # says - "yes", "no", or a correction. That is why the resume value
            # is the raw user text rather than a parsed boolean.
            payload = Command(resume=job["message"]) if resuming else state
            with otel.span(
                "chat.turn",
                kind="CHAIN",
                **{
                    otel.INPUT_VALUE: job["message"][:2000],
                    otel.USER_ID: job["user_id"],
                    otel.SESSION_ID: job.get("conversation_id", ""),
                    otel.LLM_PROVIDER: self.settings.llm_provider,
                    otel.LLM_MODEL: self.settings.llm_model,
                },
            ):
                trace_id = otel.current_trace_id()
                result = await asyncio.wait_for(
                    self.graph.ainvoke(payload, config),
                    timeout=self.settings.run_timeout_s,
                )
        except asyncio.TimeoutError:
            log.warning(
                "run %s exceeded %.0fs; on a rate-limited tier this is usually the "
                "egress token budget holding longer than the run timeout",
                run_id, self.settings.run_timeout_s,
            )
            await self._emit(run_id, "error", seq, message="The assistant timed out.")
            raise RunTimeout(run_id) from None
        except Exception as exc:
            # Log the real cause. The event carries a message safe to show a
            # policyholder, which meant a 502 arrived with nothing in the logs
            # to explain it - undiagnosable from the outside.
            log.exception("run %s failed: %s", run_id, exc)
            await self._emit(
                run_id, "error", seq,
                message="The assistant is temporarily unavailable.",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise

        seq = await self._emit_tool_calls(run_id, result, seq)

        # A pending confirmation surfaces as its own event so every surface can
        # render it: a prompt in chat, a spoken readback over voice.
        interrupts = result.get("__interrupt__") or ()
        if interrupts:
            value = interrupts[0].value
            await self._emit(
                run_id, "confirm", seq,
                tool="submit_claim",
                args=value.get("args", {}),
                readback=value.get("readback", ""),
            )
            seq += 1
            # Emitted before returning, because a turn can search the policy and
            # then propose a write: "is a burst pipe covered, and file a $1,200
            # claim on POL-1092". Returning early here dropped the search and
            # its citation from the response entirely.
            await self._emit(run_id, "sources", seq, sources=result.get("sources", []))
            seq += 1
            await self._emit(
                run_id, "done", seq, latency_ms=self._ms(started), trace_id=trace_id
            )
            return

        final = next(
            (m for m in reversed(result.get("messages", []))
             if m.type == "ai" and not getattr(m, "tool_calls", None)),
            None,
        )
        text = str(final.content) if final else ""
        if text:
            await self._emit(run_id, "token", seq, text=text)
            seq += 1

        await self._emit(run_id, "sources", seq, sources=result.get("sources", []))
        seq += 1
        await self._emit(
            run_id, "done", seq,
            latency_ms=self._ms(started),
            trace_id=trace_id,
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            stopped_reason=result.get("stopped_reason"),
        )

    async def _emit_tool_calls(self, run_id: str, result: dict[str, Any], seq: int) -> int:
        for call in result.get("tool_invocations", []):
            await self._emit(run_id, "tool_end", seq, **call)
            seq += 1
        return seq

    @staticmethod
    def _ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------- loop

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            with contextlib.suppress(Exception):
                await self.queue.reclaim_stalled(CONSUMER)

            jobs = await self.queue.consume(CONSUMER, block_ms=5_000)
            for entry_id, job in jobs:
                try:
                    await self.handle(job)
                    await self.queue.ack(entry_id)
                except Exception as exc:
                    await self.queue.dead_letter(entry_id, job, f"{type(exc).__name__}: {exc}")

    def stop(self) -> None:
        self._running = False


def seed_claims(settings: AgentSettings) -> None:
    """Copy the fixture onto the claims volume if it is not there yet.

    Never overwrites: once a policyholder has filed a claim, that file is the
    system of record, and a restart must not silently discard it.
    """
    import shutil
    from pathlib import Path

    target = Path(settings.claims_path)
    seed = Path(settings.claims_seed_path)

    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if seed.exists():
        shutil.copyfile(seed, target)
        log.info("seeded claims store from %s", seed)
    else:
        target.write_text("[]", encoding="utf-8")
        log.warning("no claims seed at %s; starting with an empty store", seed)


def build_claims_repo(settings: AgentSettings) -> Any:
    """Choose the claims store from configuration.

    The whole point of the port, and the one place that knows which adapter is
    in play. `AGENT_CLAIMS_BACKEND=postgres` moves the system of record from a
    file to a table and nothing else changes - not the tools, not the graph,
    not the API response.
    """
    from libs.adapters.claims_json import JsonFileClaimsRepo
    from libs.adapters.claims_memory import InMemoryClaimsRepo

    if settings.claims_backend == "memory":
        return InMemoryClaimsRepo()

    if settings.claims_backend == "postgres":
        from libs.adapters.claims_postgres import PostgresClaimsRepo

        if not settings.database_url:
            raise ValueError(
                "AGENT_CLAIMS_BACKEND=postgres needs AGENT_DATABASE_URL"
            )
        log.info("claims store: postgres")
        return PostgresClaimsRepo(settings.database_url)

    seed_claims(settings)
    log.info("claims store: %s", settings.claims_path)
    return JsonFileClaimsRepo(settings.claims_path)


def build_worker(
    settings: AgentSettings | None = None, checkpointer: Any | None = None
) -> AgentWorker:
    """Assemble the worker.

    `checkpointer` is injected rather than constructed here because the
    Postgres saver is an async context manager whose lifetime must match the
    worker's - see `main`.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from libs.adapters.queue_redis import RedisQueue
    from libs.adapters.queue_redis import RedisIdempotencyStore
    from .providers.registry import build_chat_model, default_retry_policy, resolve
    from .providers.resilient import ResilientChatModel
    from .retrieval_client import RetrievalClient
    from .tools.claims import build_claims_tools
    from .tools.policy import build_policy_tool

    settings = settings or AgentSettings()
    repo = build_claims_repo(settings)
    idempotency = RedisIdempotencyStore(settings.redis_url) if settings.redis_url else None
    # One client for both: the policy tool searches it, and the claims tools
    # read the whole document from it to find the cap that governs a claim.
    retrieval = RetrievalClient(settings.retrieval_url)
    tools = [
        build_policy_tool(retrieval),
        *build_claims_tools(repo, idempotency, sections=retrieval.sections),
    ]

    try:
        config = resolve(
            settings.llm_provider,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_base_url,
        )
    except ValueError as exc:
        # `docker compose up` on a fresh clone has no .env and therefore no key.
        # Exiting there would mean the brief's "docker-compose up launches
        # everything" is only true for someone who already has credentials, so
        # fall back to the keyless demo provider instead - loudly, and with
        # every answer it produces prefixed "(demo mode - no LLM configured)",
        # so nobody can mistake it for the real model.
        log.warning("%s", exc)
        log.warning(
            "falling back to the keyless demo provider. Put GROQ_API_KEY in .env "
            "(or set LLM_PROVIDER=ollama) for real answers."
        )
        config = resolve("fake")
    primary = build_chat_model(config, timeout=settings.llm_timeout_s)

    fallback = None
    if settings.llm_fallback_provider:
        fallback_config = resolve(
            settings.llm_fallback_provider, "", settings.llm_fallback_api_key
        )
        fallback = build_chat_model(fallback_config, timeout=settings.llm_timeout_s)

    # Egress limit, retry, breaker and fallback - all of which existed in
    # libs.resilience and were previously called from nowhere.
    llm = ResilientChatModel(
        primary,
        primary_name=settings.llm_provider,
        fallback=fallback,
        fallback_name=settings.llm_fallback_provider or "none",
        policy=default_retry_policy("text"),
        requests_per_minute=settings.llm_requests_per_minute,
        tokens_per_minute=settings.llm_tokens_per_minute,
    )

    graph = build_graph(
        llm,
        tools,
        checkpointer=checkpointer or InMemorySaver(),
        require_confirmation=settings.require_claim_confirmation,
        max_iterations=settings.max_graph_iterations,
    )
    return AgentWorker(settings, RedisQueue(settings.redis_url), graph)


@contextlib.asynccontextmanager
async def open_checkpointer(settings: AgentSettings):
    """Yield a checkpointer whose lifetime matches the worker's.

    `AsyncPostgresSaver.from_conn_string` returns an async context manager, not
    a saver - entering it is what opens the connection pool and lets `setup()`
    create the checkpoint tables. Getting this wrong fails loudly at graph
    compile time, which is how it was caught.

    A Postgres that is unreachable degrades to in-memory rather than refusing
    to start: only the cross-replica resume is lost, and answering is more
    valuable than that.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    if not settings.database_url:
        yield InMemorySaver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            await saver.setup()
            log.info("using Postgres checkpointer - resume works across replicas")
            yield saver
    except Exception as exc:
        log.warning(
            "Postgres checkpointer unavailable (%s); falling back to in-memory. "
            "Confirmation resume will not survive a restart or reach another replica.",
            exc,
        )
        yield InMemorySaver()


async def main() -> None:
    settings = AgentSettings()
    if otel.setup("omnicare-agent"):
        otel.instrument_langchain()
    async with open_checkpointer(settings) as checkpointer:
        try:
            worker = build_worker(settings, checkpointer)
        except ValueError as exc:
            # Configuration, not a crash. A stack trace tells the operator
            # nothing they can act on.
            log.error("agent cannot start: %s", exc)
            raise SystemExit(2) from None
        log.info("agent worker ready, consuming as %s", CONSUMER)
        await worker.run_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(main())
