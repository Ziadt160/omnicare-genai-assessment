# 0004 - Postgres for durable state; Redis stays ephemeral

**Status:** accepted · **Date:** 2026-09-03

## Context

The brief does not ask for a database. But the chat UI must show message
history, and a multi-replica agent needs a shared checkpointer for the
confirmation flow to survive.

## Decision

Postgres holds `app.conversations`, `app.messages` and the LangGraph
checkpoints. Redis keeps only ephemeral state: streams, rate limits,
idempotency keys.

## Consequences

- The `interrupt()` flow works across replicas: the follow-up yes can land on a
  different container than the one that paused. Without this, the scaling story
  in 0001 is incomplete.
- Redis has one job description instead of four, which makes the architecture
  legible.
- An insurance assistant that cannot reproduce what it told a policyholder,
  with citations and tool calls, is a compliance problem rather than a missing
  feature. That is the domain-grounded reason, and it is the strongest one.
- Cost: one container, migrations, an async driver.

## Rejected

- **Redis as system of record.** Works technically; a weak look for claims and
  conversation history in a financial context.
- **SQLite.** Retained as the documented day-3 fallback: same ports, one env
  var, one fewer container. Under multiple agent replicas it contends on locks.

## Ownership

`gateway` owns the `app` schema, so history stays readable while the agent is
down. `agent` owns the `langgraph` schema. One instance, two schemas.
Production would separate the instances.
