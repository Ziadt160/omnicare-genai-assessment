# 0007 - Confirm parsed entities, not raw transcripts

**Status:** accepted · **Date:** 2026-09-03

## Context

STT mishears identifiers. POL-1092 comes back as "pol ten ninety two"; a claim
id becomes "claim eighty-eight twenty-one". Acting on a bad transcript when
filing a claim with a dollar amount is a real-world harm.

## Decision

Do **not** confirm the raw transcript. Confirm the **parsed arguments**, which
catches transcription errors and model extraction errors in one gate, and which
is a gate the graph already has. Tier by risk:

| Tier | Trigger | Behaviour | Extra turns |
|---|---|---|---|
| 0 | Read-only, no entities | Answer directly | 0 |
| 1 | Read-only with an entity | Implicit phonetic echo mid-answer; barge-in carries the correction | 0 |
| 2 | Any write | Explicit readback and yes/no, via `interrupt()` | 1 |

Tier 1 escalates to Tier 2 when `stt_confidence` falls below threshold.

## Consequences

- Confirming a transcript every turn would be miserable UX and would still miss
  the failure that matters: a perfectly transcribed POL-1092 extracted as
  POL-1029.
- **The tier-1 read-back is generated in code, not prompted.** This was learned
  the hard way: the graph computed the tier and nothing acted on it, so the
  read-back never happened at all. Adding a voice instruction to the system
  prompt fixed that but only about half the time - qwen2.5 says "CLM-eight
  eight twenty-one", which is exactly the ambiguity the read-back exists to
  remove. A `readback` node now prefixes the answer from
  `phonetic_readback()`. Same principle as the rest of the envelope: the model
  decides what to say, deterministic code decides what must be said.
- Three supports matter more than the confirmation itself: a spoken-form
  normalizer (`libs/guardrails/normalize.py`, 23 test cases), phonetic readback
  so an id can be verified by ear, and fuzzy recovery against known ids so a
  mishearing becomes "did you mean CLM-8821?" instead of a dead end.
- The live transcript streams to the DOM over a LiveKit data channel. Visual
  confirmation costs nothing and consumes no turn.

## Rejected

- **Full transcript confirmation.** Noisy, doubles every turn, and misses
  extraction errors entirely.
- **No confirmation on reads.** Fine for Tier 0, not for an entity the agent is
  about to look up and report on.
