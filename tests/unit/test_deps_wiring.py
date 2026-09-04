"""Adapter selection.

Regression: `deps.py` imported `conversations_postgres`, which did not exist.
Because compose sets DATABASE_URL, the gateway crashed on boot in Docker while
every test passed - the in-memory branch was the only one ever exercised. This
file imports each branch so a missing adapter fails here, not on `compose up`.
"""

from __future__ import annotations

import pytest

from gateway.app.deps import build_deps
from gateway.app.settings import GatewaySettings

DSN = "postgresql://omnicare:omnicare@localhost:5432/omnicare"


def test_memory_branch() -> None:
    deps = build_deps(GatewaySettings(redis_url="", database_url=""))
    assert type(deps.queue).__name__ == "InMemoryQueue"
    assert type(deps.conversations).__name__ == "InMemoryConversationRepo"


def test_postgres_branch_is_importable() -> None:
    """Constructing must not require a reachable database - the pool opens
    lazily so the gateway can start and answer /health before Postgres is up."""
    deps = build_deps(GatewaySettings(redis_url="", database_url=DSN))
    assert type(deps.conversations).__name__ == "PostgresConversationRepo"


def test_redis_branch_is_importable() -> None:
    deps = build_deps(GatewaySettings(redis_url="redis://localhost:6379/0", database_url=""))
    assert type(deps.queue).__name__ == "RedisQueue"
    assert type(deps.rate_limiter).__name__ == "RedisRateLimiter"


@pytest.mark.parametrize(
    "settings",
    [
        GatewaySettings(redis_url="", database_url=""),
        GatewaySettings(redis_url="redis://localhost:6379/0", database_url=DSN),
    ],
    ids=["memory", "full"],
)
def test_every_adapter_satisfies_its_port(settings) -> None:
    deps = build_deps(settings)
    for name in ("enqueue", "depth", "publish", "subscribe"):
        assert hasattr(deps.queue, name), f"queue missing {name}"
    for name in ("ensure", "add_message", "list_for_user", "messages"):
        assert hasattr(deps.conversations, name), f"conversations missing {name}"
    assert hasattr(deps.rate_limiter, "check")


# ---------------------------------------------------- claims store selection

@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("memory", "InMemoryClaimsRepo"),
        ("json", "JsonFileClaimsRepo"),
        ("postgres", "PostgresClaimsRepo"),
    ],
)
def test_every_claims_backend_is_selectable(backend, expected, tmp_path) -> None:
    """The JSON-to-Postgres swap is one environment variable.

    `claims_backend` offered "postgres" before any such adapter existed, so the
    setting promised a swap the code could not perform. All three now resolve.
    """
    from agent.app.settings import AgentSettings
    from agent.app.worker import build_claims_repo

    seed = tmp_path / "seed.json"
    seed.write_text("[]", encoding="utf-8")
    repo = build_claims_repo(
        AgentSettings(
            claims_backend=backend,
            claims_path=str(tmp_path / "claims.json"),
            claims_seed_path=str(seed),
            database_url=DSN,
        )
    )
    assert type(repo).__name__ == expected


def test_all_claims_adapters_satisfy_the_port(tmp_path) -> None:
    from agent.app.settings import AgentSettings
    from agent.app.worker import build_claims_repo

    seed = tmp_path / "seed.json"
    seed.write_text("[]", encoding="utf-8")
    for backend in ("memory", "json", "postgres"):
        repo = build_claims_repo(
            AgentSettings(
                claims_backend=backend,
                claims_path=str(tmp_path / f"{backend}.json"),
                claims_seed_path=str(seed),
                database_url=DSN,
            )
        )
        for method in ("get", "append", "list_ids"):
            assert hasattr(repo, method), f"{backend} missing {method}"


def test_postgres_backend_without_a_dsn_fails_loudly() -> None:
    """Silently falling back to a file would put claims somewhere the operator
    did not ask for."""
    from agent.app.settings import AgentSettings
    from agent.app.worker import build_claims_repo

    with pytest.raises(ValueError, match="AGENT_DATABASE_URL"):
        build_claims_repo(AgentSettings(claims_backend="postgres", database_url=""))
