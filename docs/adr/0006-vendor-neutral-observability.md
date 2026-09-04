# 0006 - Vendor-neutral tracing; Phoenix locally

**Status:** accepted · **Date:** 2026-09-03

## Context

Agent behaviour needs to be inspectable: which tool was chosen, what was
retrieved, why an answer was wrong. LangSmith is the obvious tool but is
hosted, and the brief is zero-cost and local-first.

## Decision

Instrument with **OpenTelemetry plus OpenInference semantic conventions**, not
a vendor SDK. The backend is one environment variable:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006/v1/traces
```

Arize Phoenix runs locally under `--profile obs`. Unset means a no-op exporter.

## Consequences

- Tracing is never a hard dependency; the system runs correctly with it off.
- `trace_id` is stored on the `messages` row and returned in a response header,
  so any message in history links straight to its trace. That is what makes
  tracing useful rather than a screenshot.
- Deterministic pytest evals remain the CI gate. Phoenix carries exploration
  and non-blocking quality scores. A build that cannot pass without a tracing
  UI running has a fragile gate.

## Rejected

- **Langfuse self-hosted.** The closest LangSmith equivalent, but current
  self-host wants ClickHouse, MinIO, its own Postgres and Redis: four or five
  containers on top of eight. Named in the README as the production choice.
- **LangSmith.** Supported by the same exporter, but data leaves the machine.
- **Jaeger or Tempo.** Generic traces without prompt and completion rendering.
