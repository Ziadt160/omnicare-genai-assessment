# 0005 - Ports and adapters at every I/O boundary

**Status:** accepted · **Date:** 2026-09-03

## Context

The brief says `submit_claim` appends to `mock_claims.json`. That should not
become a permanent architectural constraint, and TDD is impossible if tests
need real infrastructure.

## Decision

**Every I/O boundary is a `Protocol` in `libs/ports` with an in-memory adapter.**

| Port | Production | Test |
|---|---|---|
| `ClaimsRepository` | `JsonFile` (default), `Postgres` | `InMemory` |
| `ConversationRepository` | `Postgres` | `InMemory` |
| `VectorStore` | `Qdrant` | `InMemory` |
| `LLMProvider` | `OpenAICompatible` | `FakeLLM` |
| `JobQueue` | `RedisStreams` | `InMemory` |
| `RateLimiter` | `Redis` | `InMemory` |

Three rules make these real repositories rather than thin wrappers:

1. Async interfaces returning **domain models**, never driver rows.
2. **Domain exceptions** such as `ClaimNotFound` and `DuplicateClaim`, with no
   driver error leaking upward.
3. Adapter chosen by a **factory reading `Settings`**, so no backend branching
   appears at call sites.

## Consequences

- Unit tests run in under a second with zero containers, which is what makes
  test-first viable inside the timebox.
- JSON to Postgres is one env var, `AGENT_CLAIMS_BACKEND`.
- The JSON adapter must be genuinely concurrency-safe, since the agent scales:
  `flock` on a shared mount (the same inode across containers) plus
  write-to-temp and `os.replace`. Asserted by
  `test_concurrent_appends_do_not_corrupt_or_drop`.

## Rejected

- **Direct driver calls.** Fast to write, impossible to test without Docker.
- **ORM models as domain models.** Couples the database schema to the wire
  contract, so a migration becomes an API change.
