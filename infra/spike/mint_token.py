"""Mint a LiveKit dev JWT for the day-1 WebRTC spike.

Standalone on purpose: no project imports, no dependencies beyond PyJWT, so the
gate can be run before any service exists. The real gateway mints these at
POST /api/v1/voice/token and the secret never reaches the browser.

    python infra/spike/mint_token.py
    # then open the printed URL

Named mint_token.py, not token.py: a module called `token` shadows the standard
library's, and anything in this directory that imports `inspect` or `tokenize`
- `python -m http.server` among them - fails with a confusing ImportError.

If the ICE state in the page reaches `connected`, WebRTC works through Docker
on this machine. If it hangs at `checking`, LiveKit is advertising an IP the
browser cannot reach - see infra/livekit.yaml.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import webbrowser
from pathlib import Path

# Matches infra/livekit.yaml. DEV ONLY.
API_KEY = "devkey"
API_SECRET = "devsecret-local-only-not-for-production"
ROOM = "spike"
IDENTITY = "spike-user"
TTL_SECONDS = 3600


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def mint(api_key: str, api_secret: str, room: str, identity: str, ttl: int) -> str:
    """Build a LiveKit access token (HS256 JWT) without the SDK.

    LiveKit expects the grants under a ``video`` claim, ``iss`` set to the API
    key, and ``sub`` to the participant identity.
    """
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": api_key,
        "sub": identity,
        "name": identity,
        "nbf": now - 10,
        "exp": now + ttl,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    signature = hmac.new(
        api_secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64(signature)}"


if __name__ == "__main__":
    token = mint(API_KEY, API_SECRET, ROOM, IDENTITY, TTL_SECONDS)
    page = Path(__file__).with_name("index.html").resolve()
    url = page.as_uri() + "?token=" + token

    print("token minted for room", ROOM, "\n")
    print(url, "\n")
    print("Start LiveKit first:")
    print("  docker compose up livekit\n")
    print("Then open the URL above and click 'Connect and publish mic'.")
    print("Watch for 'ICE CONNECTED' (gate passed) or 'ICE stuck' (gate failed).")

    try:
        webbrowser.open(url)
    except Exception:
        pass
