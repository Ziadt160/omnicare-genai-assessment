"""Gateway settings.

One BaseSettings class per service with its own env prefix - that is what
"decoupled settings" means in practice: the gateway can be reconfigured without
touching the agent, and neither can read the other's variables by accident.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = ""

    # Ingress protection. The egress limiter lives in the agent, against the
    # provider's RPM - conflating the two is the classic mistake.
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 10

    # Admission control: refuse rather than accept work that cannot finish in
    # time. A queue that accepts everything is a slower timeout.
    max_queue_depth: int = 50

    # Must be >= the agent's own run timeout, or the gateway gives up while the
    # agent is still working - throwing away the provider quota it just spent
    # waiting out a rate limit. Restart gateway and agent together, with the
    # same env file; `make up-groq` and `make up-ollama` do.
    run_timeout_s: float = 30.0

    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "devsecret-local-only-not-for-production"
    livekit_token_ttl_s: int = 900

    cors_origins: str = "*"
