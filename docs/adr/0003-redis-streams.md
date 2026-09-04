# 0003 - Redis Streams for the queue and the token transport

**Status:** accepted · **Date:** 2026-09-03

## Context

Two needs that look separate: stream tokens to the browser in real time, and
queue work when the agent is saturated.

## Decision

Redis **Streams** for both. `XADD` enqueues onto `jobs:chat`; a consumer group
load-balances across agent replicas; `XACK` completes; `XAUTOCLAIM` recovers
jobs from a replica that died mid-run. Each run writes events to
`stream:{run_id}`, which the gateway tails and forwards.

## Consequences

- The queue costs **zero** new containers.
- A dropped WebSocket resumes from the last delivered entry id rather than
  losing the answer.
- The graded synchronous `POST /api/v1/chat` is preserved: the gateway blocks
  on the result stream while the job waits in the job stream.
- Admission control (`XLEN` plus pending depth) returns 429 rather than
  accepting work that cannot finish inside the deadline.

## Rejected

- **Redis Pub/Sub.** Fire-and-forget; a gateway restart mid-answer loses tokens.
- **Celery.** Sync-first, awkward with async LangGraph, plus worker and beat
  containers.
- **Kafka or RabbitMQ.** Overkill, slow to start, heavy for the timebox.
- **NATS JetStream.** Genuinely good and lighter, but Redis is already required
  for rate limits and idempotency, so it would be a net addition.
