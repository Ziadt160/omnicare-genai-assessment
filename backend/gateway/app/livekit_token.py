"""LiveKit access tokens, minted server-side only.

Hand-rolled HS256 rather than pulling the LiveKit server SDK: it is thirty
lines, it removes a dependency from the gateway image, and it makes the claim
structure visible in review - which matters, because getting the `video` grant
wrong produces a token that connects and then silently cannot publish.

The API secret never leaves this process. infra/spike/mint_token.py duplicates this
deliberately: the day-1 gate must run before any service exists.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def mint_access_token(
    *,
    api_key: str,
    api_secret: str,
    room: str,
    identity: str,
    ttl_s: int = 900,
    can_publish: bool = True,
    can_subscribe: bool = True,
) -> str:
    """Build a LiveKit JWT.

    Args:
        room: Scoped to exactly one room. A token that can join any room is a
            token that can join someone else's conversation.
        ttl_s: Short by default. The browser asks for a new one per session.
    """
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": api_key,
        "sub": identity,
        "name": identity,
        "nbf": now - 10,
        "exp": now + ttl_s,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": can_publish,
            "canSubscribe": can_subscribe,
            "canPublishData": True,
        },
    }
    signing_input = (
        f"{_b64(json.dumps(header, separators=(',', ':')).encode())}."
        f"{_b64(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    signature = hmac.new(
        api_secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64(signature)}"
