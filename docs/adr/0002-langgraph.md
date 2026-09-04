# 0002 - LangGraph, single ReAct loop

**Status:** accepted · **Date:** 2026-09-03

## Context

The workflow is: answer coverage questions from a document, look up a claim,
file a claim. Two operational tools plus retrieval.

## Decision

A single ReAct loop in LangGraph, wrapped in deterministic nodes:

```
guard -> agent <-> tools -> confirm -> ground -> END
```

`guard` and `ground` contain no LLM call. The model chooses tools; it does not
decide whether input was safe or whether a citation was real. Retrieval is
exposed as a third tool so the graph stays one clean loop instead of a router.

## Consequences

- `interrupt()` before `submit_claim` is a first-class graph feature, not
  prompt discipline. With a Postgres checkpointer it survives restarts and
  works across replicas.
- `MAX_GRAPH_ITERATIONS=5` is enforced structurally. Without it, a weak model
  that cannot find a claim loops on the same tool and burns free-tier quota.
- One graph serves REST, WebSocket and voice, so it is evaluated once.

## Rejected

- **CrewAI and supervisor topologies.** Multi-agent orchestration for two tools
  adds latency and failure modes with nothing to show for it, and reads as
  miscalibration rather than sophistication.
- **LangChain alone.** Good tool abstractions, but no checkpointed,
  interruptible state machine, which is the entire safety argument here.
- **A hand-rolled loop.** Would mean reimplementing checkpointing and interrupts.
