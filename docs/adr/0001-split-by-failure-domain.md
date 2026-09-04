# 0001 - Split by failure domain, not by entity

**Status:** accepted · **Date:** 2026-09-03

## Context

A three-day prototype with a graded "two-minute build and run" requirement. The
brief asks only for `/frontend` and `/backend` separation. Splitting further
buys fault isolation and independent scaling; it costs containers, network
hops, and build time.

## Decision

Three services and one worker. Every boundary is justified by a **different**
constraint - that is the test for whether a split is real:

| Service | Constraint |
|---|---|
| `gateway` | Owns the public contract; must answer `/health` and return a clean 503 while the agent is down. |
| `agent` | The flaky tier - external LLM calls, rate limits, timeouts. Stateless, so it scales horizontally. |
| `retrieval` | Embedding model in RAM with a slow warm start; in-process it would reload on every agent restart. |
| `voice` | A LiveKit dispatch worker, not an HTTP server. Different runtime model and heavy audio dependencies. |

Claims are a repository inside the agent, **not** a service - see 0005.

## Consequences

- The scaling claim is demonstrable: `docker compose up --scale agent=4`.
- Eight containers is heavy for the timebox. Mitigations are load-bearing, not
  cosmetic: `fastembed` over PyTorch, a shared Python base image, `uv`, Ollama
  out of the default profile, and voice that degrades rather than blocking chat.
- Drift between services is contained by `libs/contracts`, imported by all.

## Rejected

- **Modular monolith.** Simpler and defensible, but gives up the fault
  isolation that motivated the exercise.
- **A service per noun** (claims, conversations, policies). Boundaries with no
  distinct constraint behind them, which is a distributed monolith.
