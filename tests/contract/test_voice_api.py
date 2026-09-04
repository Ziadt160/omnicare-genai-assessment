"""`POST /api/v1/voice/token`, and the conversation identity it carries.

The interesting property here is not the JWT - it is that a voice call and a
typed conversation are **the same conversation**. The agent keys its memory on
`conversation_id`, so if the room encodes a different one, a caller who has just
been typing has to introduce themselves again, and a confirmation paused in chat
cannot be resumed by voice.

The room name is the only channel the worker has for this: it is dispatched into
a room and told nothing else. So the room *is* the conversation id, wrapped in a
prefix, and these tests pin the round trip from both ends.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from gateway.app.deps import Deps
from gateway.app.main import app
from gateway.app.settings import GatewaySettings
from libs.adapters.conversations_memory import InMemoryConversationRepo
from libs.adapters.queue_memory import InMemoryQueue, InMemoryRateLimiter
from libs.contracts.voice import conversation_id_from_room, room_for

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture()
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app.state.deps = Deps(
        settings=GatewaySettings(redis_url="", database_url="", run_timeout_s=3.0),
        queue=InMemoryQueue(),
        conversations=InMemoryConversationRepo(),
        rate_limiter=InMemoryRateLimiter(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def mint(client: httpx.AsyncClient, **body) -> dict:
    response = await client.post("/api/v1/voice/token", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ------------------------------------------------------- the graded basics

async def test_the_secret_never_reaches_the_browser(client) -> None:
    body = await mint(client, user_id="u1")
    assert body["token"].count(".") == 2, "a JWT has three segments"
    assert "secret" not in str(body).lower()


async def test_unknown_keys_are_rejected(client) -> None:
    response = await client.post(
        "/api/v1/voice/token", json={"user_id": "u1", "admin": True}
    )
    assert response.status_code == 422


# --------------------------------------------- one conversation, two channels

async def test_the_response_names_the_conversation(client) -> None:
    """The browser needs the id back: it may have had none to send, and every
    later typed turn has to land on the same thread the call used."""
    body = await mint(client, user_id="u1")
    assert body["conversation_id"], "the caller must be told which conversation this is"


async def test_an_existing_conversation_is_joined_not_replaced(client) -> None:
    """Type first, then press the mic: the call must continue that thread."""
    body = await mint(client, user_id="u1", conversation_id="cnv_alreadytyping")
    assert body["conversation_id"] == "cnv_alreadytyping"
    assert conversation_id_from_room(body["room"]) == "cnv_alreadytyping"


async def test_a_fresh_caller_gets_a_conversation_to_keep(client) -> None:
    """Speak first, then type: the browser adopts the id minted here, so the
    typed turn joins the call's thread rather than starting a second one."""
    body = await mint(client, user_id="u1")
    assert conversation_id_from_room(body["room"]) == body["conversation_id"]


async def test_two_calls_by_one_user_share_a_thread(client) -> None:
    first = await mint(client, user_id="u1")
    second = await mint(client, user_id="u1", conversation_id=first["conversation_id"])
    assert second["room"] == first["room"]


# ----------------------------------------------------------- the round trip

@pytest.mark.parametrize(
    "conversation_id",
    ["cnv_abc123", "550e8400-e29b-41d4-a716-446655440000", "usr_deadbeef"],
)
async def test_room_encoding_survives_the_trip_to_the_worker(conversation_id) -> None:
    """The worker recovers the id from the room name and nothing else.

    Postgres mints UUIDs and the in-memory repo mints `cnv_`-prefixed ids, so
    hyphens have to survive - a naive `split("-")` would truncate a UUID to
    "550e8400" and quietly give every call its own memory.
    """
    assert conversation_id_from_room(room_for(conversation_id)) == conversation_id


async def test_a_room_without_the_prefix_is_returned_unchanged() -> None:
    """A room created by hand, or by an older client, must still be usable as a
    thread id rather than raising in the middle of a call."""
    assert conversation_id_from_room("some-other-room") == "some-other-room"
