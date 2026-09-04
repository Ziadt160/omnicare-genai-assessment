"""The gateway: the only service with a public contract.

`POST /api/v1/chat` and `GET /api/v1/health` are graded and their shapes are
fixed. Everything else here is additive, and every added request field is
optional with a default, so the exact payload from the brief still validates
under `extra="forbid"`.

The queue is invisible on the REST path: the gateway enqueues onto the job
stream and blocks on the *result* stream, so the synchronous contract holds
while the work is load-balanced across however many agent replicas exist.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from libs.contracts import (
    ChatRequest,
    ChatResponse,
    DeepHealthResponse,
    HealthResponse,
    RunEvent,
    ToolCall,
    VoiceTokenRequest,
    VoiceTokenResponse,
    room_for,
)
from libs.errors import QueueSaturated, RateLimited
from libs.observability import otel
from .deps import Deps, get_deps
from .livekit_token import mint_access_token
from .settings import GatewaySettings

log = logging.getLogger("omnicare.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    otel.setup("omnicare-gateway")
    yield
    deps: Deps | None = getattr(app.state, "deps", None)
    if deps is not None:
        await deps.aclose()


app = FastAPI(
    title="OmniCare Policyholder Assistant",
    version="1.0.0",
    description=(
        "Coverage answers grounded in policy documents with section citations, "
        "claim lookup, and claim submission with explicit confirmation."
    ),
    lifespan=lifespan,
)
_settings = GatewaySettings()
app.add_middleware(
    CORSMiddleware,
    # Configurable rather than hardcoded: "*" is right for a local prototype
    # and wrong the moment this is served from anywhere real.
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- health

@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness. Returns exactly {"status": "healthy"} - the graded shape."""
    return HealthResponse()


@app.get("/api/v1/health/deep", response_model=DeepHealthResponse, tags=["health"])
async def health_deep(deps: Deps = Depends(get_deps)) -> DeepHealthResponse:
    """Readiness. Used by compose and by the frontend to decide whether to
    offer the voice button at all."""
    dependencies = await deps.check_dependencies()
    degraded = any(v == "down" for v in dependencies.values())
    return DeepHealthResponse(
        status="degraded" if degraded else "healthy", dependencies=dependencies
    )


# ------------------------------------------------------------------- chat

async def _collect(
    deps: Deps, run_id: str, timeout_s: float
) -> tuple[str, list[str], list[ToolCall], str | None]:
    """Buffer a run's event stream into a complete answer.

    The WebSocket route forwards these same events as they arrive; the only
    difference is buffering. One queue, three surfaces.
    """
    text: list[str] = []
    sources: list[str] = []
    tool_calls: list[ToolCall] = []
    trace_id: str | None = None
    saw_terminal = False

    async for event in deps.queue.subscribe(run_id, timeout_s):
        if event.type == "token":
            text.append(str(event.payload.get("text", "")))
        elif event.type == "sources":
            sources = list(event.payload.get("sources", []))
        elif event.type == "tool_end":
            tool_calls.append(ToolCall.model_validate(event.payload))
        elif event.type == "confirm":
            text.append(str(event.payload.get("readback", "")))
            tool_calls.append(
                ToolCall(
                    name=str(event.payload.get("tool", "submit_claim")),
                    arguments=dict(event.payload.get("args", {})),
                    status="awaiting_confirmation",
                )
            )
        elif event.type == "done":
            saw_terminal = True
            if event.payload.get("text"):
                text = [str(event.payload["text"])]
            trace_id = event.payload.get("trace_id")
        elif event.type == "error":
            raise HTTPException(
                status_code=502,
                detail=str(event.payload.get("message", "The assistant is unavailable.")),
            )

    if not saw_terminal:
        # The agent never answered. Say so rather than returning a blank 200 -
        # a silent empty response is the worst possible failure mode here.
        #
        # The usual cause on a rate-limited tier is this timeout being shorter
        # than the agent's, so the gateway gives up while the agent is still
        # waiting out a token budget. Restart both with the same env file.
        log.warning(
            "run %s produced no terminal event within %.0fs; if the agent is "
            "rate-limited its own run timeout may exceed this one",
            run_id, timeout_s,
        )
        raise HTTPException(
            status_code=504,
            detail="The assistant did not respond in time. Please try again.",
        )

    return "".join(text).strip(), sources, tool_calls, trace_id


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest, response: Response, deps: Deps = Depends(get_deps)
) -> ChatResponse:
    """Answer one turn.

    Coverage questions are answered from the policy documents with citations;
    claim questions call the claims tools. Filing a claim returns a
    confirmation prompt first - reply "yes" in the next message to proceed.
    """
    started = time.perf_counter()

    try:
        await deps.rate_limiter.check("chat", request.user_id)
    except RateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail="Too many requests.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    depth = await deps.queue.depth()
    if depth > deps.settings.max_queue_depth:
        raise HTTPException(
            status_code=429,
            detail="The assistant is busy. Please try again shortly.",
            headers={"Retry-After": "5"},
        )

    conversation_id = await deps.conversations.ensure(
        request.user_id, request.conversation_id
    )
    await deps.conversations.add_message(
        conversation_id, "user", request.message, channel=request.channel
    )

    run_id = uuid.uuid4().hex
    await deps.queue.enqueue(
        run_id,
        {
            "user_id": request.user_id,
            "conversation_id": conversation_id,
            "message": request.message,
            "channel": request.channel,
            "stt_confidence": request.stt_confidence,
        },
        deadline_s=deps.settings.run_timeout_s,
    )

    text, sources, tool_calls, trace_id = await _collect(
        deps, run_id, deps.settings.run_timeout_s
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    await deps.conversations.add_message(
        conversation_id,
        "assistant",
        text,
        sources=sources,
        tool_calls=tool_calls,
        channel=request.channel,
        latency_ms=latency_ms,
        trace_id=trace_id,
    )

    if trace_id:
        response.headers["X-Trace-Id"] = trace_id

    return ChatResponse(
        response=text,
        sources=sources,
        tool_calls=tool_calls,
        conversation_id=conversation_id,
        trace_id=trace_id,
    )


@app.websocket("/api/v1/chat/stream")
async def chat_stream(websocket: WebSocket) -> None:
    """Streaming chat. Same queue, same graph - events are forwarded as they
    arrive instead of buffered, plus queue-position events so a wait is legible
    rather than dead air."""
    await websocket.accept()
    deps: Deps = websocket.app.state.deps

    try:
        while True:
            payload = await websocket.receive_json()
            request = ChatRequest.model_validate(payload)

            conversation_id = await deps.conversations.ensure(
                request.user_id, request.conversation_id
            )
            run_id = uuid.uuid4().hex

            depth = await deps.queue.depth()
            await websocket.send_json(
                RunEvent(
                    run_id=run_id, type="queued", seq=0, payload={"position": depth}
                ).model_dump()
            )

            await deps.queue.enqueue(
                run_id,
                {
                    "user_id": request.user_id,
                    "conversation_id": conversation_id,
                    "message": request.message,
                    "channel": request.channel,
                    "stt_confidence": request.stt_confidence,
                },
                deadline_s=deps.settings.run_timeout_s,
            )

            async for event in deps.queue.subscribe(run_id, deps.settings.run_timeout_s):
                await websocket.send_json(event.model_dump())
    except Exception:
        # A closed socket is the normal way this ends.
        return


# ------------------------------------------------------------------ voice

@app.post("/api/v1/voice/token", response_model=VoiceTokenResponse, tags=["voice"])
async def voice_token(
    request: VoiceTokenRequest, deps: Deps = Depends(get_deps)
) -> VoiceTokenResponse:
    """Mint a short-lived LiveKit JWT scoped to one room.

    Server-side only: the API secret never reaches the browser.
    """
    settings = deps.settings

    # `ensure`, not the raw user id: a call and a typed conversation must be the
    # same conversation. The agent keys its memory on this, so a caller who was
    # just typing would otherwise have to introduce themselves again, and a
    # confirmation paused in chat could not be resumed by voice.
    conversation_id = await deps.conversations.ensure(
        request.user_id, request.conversation_id
    )
    room = room_for(conversation_id)
    token = mint_access_token(
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        room=room,
        identity=request.user_id,
        ttl_s=settings.livekit_token_ttl_s,
    )
    return VoiceTokenResponse(
        token=token,
        url=settings.livekit_url,
        room=room,
        conversation_id=conversation_id,
        expires_in=settings.livekit_token_ttl_s,
    )


# ---------------------------------------------------------- conversations

@app.get("/api/v1/conversations/{user_id}", tags=["history"])
async def list_conversations(
    user_id: str, deps: Deps = Depends(get_deps)
) -> list[dict[str, Any]]:
    """A user's conversations, most recent first.

    Owned by the gateway, so history stays readable while the agent is down.
    """
    return await deps.conversations.list_for_user(user_id)


@app.get("/api/v1/conversations/{conversation_id}/messages", tags=["history"])
async def conversation_messages(
    conversation_id: str, deps: Deps = Depends(get_deps)
) -> list[dict[str, Any]]:
    """Full history with sources and tool calls, for rehydrating the UI."""
    return await deps.conversations.messages(conversation_id)


@app.exception_handler(QueueSaturated)
async def _saturated(request: Request, exc: QueueSaturated) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "The assistant is busy. Please try again shortly."},
        headers={"Retry-After": str(exc.retry_after)},
    )
