"""Agent settings. Own env prefix, so it cannot read the gateway's variables
by accident - that is what decoupled configuration buys."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")

    # --- model ---
    llm_provider: str = "groq"
    llm_model: str = ""
    llm_api_key: str = ""
    # Overrides the provider's default endpoint. Needed for an Ollama running
    # on the host rather than in compose: the container reaches it at
    # host.docker.internal, not at the service name.
    llm_base_url: str = ""
    llm_timeout_s: float = 20.0
    llm_fallback_provider: str = ""
    llm_fallback_api_key: str = ""

    # Egress limiting, against the provider's own RPM. Distinct from the
    # gateway's ingress limiter: different window, different key, different
    # failure behaviour. Conflating them is the classic mistake.
    llm_requests_per_minute: int = 25
    # The limit that actually binds on a hosted free tier: Groq allows 1000
    # requests per day but only 8000 tokens per minute, and each call here is
    # roughly 2000 tokens - so requests are never the constraint.
    #
    # Defaults to 0 (disabled) because a local model has no such ceiling and a
    # stubbed one reports no usage, which leaves every reservation uncorrected
    # and throttles a provider that answers instantly.
    llm_tokens_per_minute: int = 0

    # --- infrastructure ---
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = ""
    retrieval_url: str = "http://localhost:8000"

    # --- claims store ---
    claims_backend: Literal["json", "postgres", "memory"] = "json"
    claims_path: str = "/data/claims/mock_claims.json"
    # Copied into claims_path on first start. The claims file lives on a named
    # volume so filed claims survive `docker compose down`, which means the
    # volume starts empty and something has to put the fixture there.
    claims_seed_path: str = "/data/seed/mock_claims.json"

    # --- graph ---
    require_claim_confirmation: bool = True
    max_graph_iterations: int = 5
    run_timeout_s: float = 25.0
