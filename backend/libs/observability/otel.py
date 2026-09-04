"""Tracing setup. Vendor-neutral by construction.

Instrumented with OpenTelemetry rather than a vendor SDK, so the backend is one
environment variable:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006/v1/traces

Point it at Phoenix locally, LangSmith, Langfuse, Jaeger - or leave it unset and
every call here becomes a no-op. That last property is the important one: a
build that cannot run without a tracing UI has a fragile dependency, so this
module never raises and never blocks, whatever the exporter is doing.

Everything is imported lazily. The OTel packages are real dependencies of the
images but not of the unit suite, and importing this module must stay free.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger("omnicare.otel")

_ENABLED = False
_TRACER: Any = None

# OpenInference semantic conventions - the same attribute names Phoenix,
# LangSmith and Langfuse all read, which is what makes the backend swappable.
SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL = "llm.model_name"
LLM_PROVIDER = "llm.provider"
TOOL_NAME = "tool.name"
SESSION_ID = "session.id"
USER_ID = "user.id"


def setup(service_name: str = "omnicare") -> bool:
    """Configure the tracer once. Returns whether tracing is actually on.

    An unreachable collector must not slow the request path, so the exporter is
    batched and its failures are logged by the SDK rather than raised here.
    """
    global _ENABLED, _TRACER

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        log.info("tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT unset)")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": os.environ.get(
                "OTEL_SERVICE_NAME", service_name
            )})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(service_name)
        _ENABLED = True
        log.info("tracing enabled, exporting to %s", endpoint)
    except Exception as exc:
        # Never fatal. Losing traces is an inconvenience; refusing to start is
        # an outage.
        log.warning("tracing setup failed (%s); continuing without it", exc)
        _ENABLED = False

    return _ENABLED


def enabled() -> bool:
    return _ENABLED


@contextmanager
def span(name: str, kind: str = "CHAIN", **attributes: Any) -> Iterator[Any]:
    """Open a span, or do nothing at all when tracing is off.

    Yields the span so a caller can set attributes, and a `None` when disabled -
    callers must tolerate that, which keeps the disabled path genuinely free
    rather than merely cheap.
    """
    if not _ENABLED or _TRACER is None:
        yield None
        return

    with _TRACER.start_as_current_span(name) as current:
        try:
            current.set_attribute(SPAN_KIND, kind)
            for key, value in attributes.items():
                if value is not None:
                    current.set_attribute(key, value)
        except Exception:
            pass
        yield current


def current_trace_id() -> str | None:
    """The active trace id as a 32-character hex string, or None.

    Stored on the `messages` row and returned as `X-Trace-Id`, so any message
    in chat history links straight to its trace. That link is what makes
    tracing useful during a demo instead of a screenshot of a dashboard.
    """
    if not _ENABLED:
        return None
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return format(context.trace_id, "032x")
    except Exception:
        return None


def instrument_langchain() -> bool:
    """Attach OpenInference auto-instrumentation to LangChain/LangGraph.

    Gives prompt, completion and tool-call spans without touching the graph
    code. Optional: a missing package disables it rather than breaking startup.
    """
    if not _ENABLED:
        return False
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
        log.info("LangChain instrumentation attached")
        return True
    except Exception as exc:
        log.info("LangChain instrumentation unavailable (%s)", exc)
        return False
