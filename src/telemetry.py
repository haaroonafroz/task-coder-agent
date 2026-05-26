"""
Arize Phoenix observability layer.

Instruments every LLM call (Orchestrator, Worker, Validator) and every tool
invocation with OpenInference-compliant OpenTelemetry spans.

.env variables consumed:
  PHOENIX_HOST      — host of the Phoenix collector (with or without http:// prefix)
  PHOENIX_PORT      — port of the Phoenix collector (default 6006)
  PHOENIX_EXTERNAL  — if "true", Phoenix is already running externally; this
                      module will NOT call px.launch_app(), only configure the
                      OTLP exporter to point at the existing instance.

Usage:
    from src.telemetry import initialize_observability, span_llm_call, span_tool_call
    initialize_observability()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Read .env values
# ---------------------------------------------------------------------------

def _strip_protocol(host: str) -> str:
    """Remove http:// or https:// prefix so the host is bare (e.g. 127.0.0.1)."""
    return host.replace("https://", "").replace("http://", "").rstrip("/")


_PHOENIX_HOST_RAW  = os.getenv("PHOENIX_HOST", "127.0.0.1")
_PHOENIX_HOST      = _strip_protocol(_PHOENIX_HOST_RAW)
_PHOENIX_PORT      = int(os.getenv("PHOENIX_PORT", "6006"))
_PHOENIX_EXTERNAL  = os.getenv("PHOENIX_EXTERNAL", "false").strip().lower() == "true"

_TRACER_NAME = "task-coder-agent"

# ---------------------------------------------------------------------------
# Optional heavy imports — gracefully degrade if packages not installed
# ---------------------------------------------------------------------------
try:
    import phoenix as px
    _PHOENIX_AVAILABLE = True
except ImportError:
    _PHOENIX_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

try:
    from openinference.instrumentation.openai import OpenAIInstrumentor
    _OPENINFERENCE_AVAILABLE = True
except ImportError:
    _OPENINFERENCE_AVAILABLE = False

_tracer_provider: Optional[object] = None


def initialize_observability(
    host: str = _PHOENIX_HOST,
    port: int = _PHOENIX_PORT,
    external: bool = _PHOENIX_EXTERNAL,
    console_fallback: bool = True,
) -> Optional[object]:
    """
    Configure OpenTelemetry to emit spans to an Arize Phoenix collector.

    Behaviour:
      - PHOENIX_EXTERNAL=true  → assumes Phoenix is already running; only
        wires the OTLP exporter to http://<host>:<port>/v1/traces.
      - PHOENIX_EXTERNAL=false → calls px.launch_app() to start a local
        Phoenix server, then wires the exporter.
      - If Phoenix / opentelemetry-sdk is not installed, falls back to
        a console span exporter (if console_fallback=True) or is a no-op.

    Spans are exported via BatchSpanProcessor (non-blocking background thread)
    rather than SimpleSpanProcessor to avoid adding OTLP round-trip latency
    to every LLM call.

    Args:
        host:             Bare hostname (no http:// prefix).
        port:             Collector port.
        external:         Skip px.launch_app() if True.
        console_fallback: Fall back to console exporter if Phoenix absent.

    Returns:
        The Phoenix session object, or None if Phoenix is not available.
    """
    global _tracer_provider

    if not _OTEL_AVAILABLE:
        print("[Telemetry] opentelemetry-sdk not installed — observability disabled.")
        return None

    # 1. Optionally launch Phoenix (only when not external)
    session = None
    if _PHOENIX_AVAILABLE and not external:
        try:
            session = px.launch_app(host=host, port=port)
            print(f"[Telemetry] Phoenix started — dashboard at http://{host}:{port}")
        except Exception as exc:
            print(f"[Telemetry] Could not start Phoenix: {exc}")
    elif external:
        print(f"[Telemetry] External Phoenix detected — connecting to http://{host}:{port}")

    # 2. Build TracerProvider with BatchSpanProcessor (non-blocking exports)
    provider = TracerProvider()
    otlp_endpoint = f"http://{host}:{port}/v1/traces"

    if _PHOENIX_AVAILABLE or external:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            print(f"[Telemetry] OTLP exporter → {otlp_endpoint} (batch mode)")
        except Exception as exc:
            print(f"[Telemetry] OTLP exporter setup failed: {exc}")
            if console_fallback:
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif console_fallback:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        print("[Telemetry] Phoenix unavailable — spans logged to console.")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    # 3. Auto-instrument OpenAI client
    if _OPENINFERENCE_AVAILABLE:
        try:
            OpenAIInstrumentor().instrument()
            print("[Telemetry] OpenAI auto-instrumentation active.")
        except Exception as exc:
            print(f"[Telemetry] OpenAI instrumentation failed: {exc}")

    return session


def get_tracer():
    """Return a named tracer for manual span creation."""
    if not _OTEL_AVAILABLE:
        return _NoOpTracer()
    return trace.get_tracer(_TRACER_NAME)


def span_llm_call(agent_role: str, milestone_id: str, model: str):
    """
    Context manager wrapping an LLM call with a labelled span.

    Usage:
        with span_llm_call("worker", "M2", "local/qwen3-27b"):
            result = call_llm(...)
    """
    return get_tracer().start_as_current_span(
        f"llm.{agent_role}",
        attributes={
            "agent.role": agent_role,
            "mission.milestone_id": milestone_id,
            "llm.model": model,
        },
    )


def span_tool_call(tool_name: str, milestone_id: str):
    """Context manager wrapping a tool execution with a labelled span."""
    return get_tracer().start_as_current_span(
        f"tool.{tool_name}",
        attributes={
            "tool.name": tool_name,
            "mission.milestone_id": milestone_id,
        },
    )


# ---------------------------------------------------------------------------
# No-op fallback tracer
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def set_attribute(self, *_):
        pass

    def record_exception(self, *_):
        pass


class _NoOpTracer:
    def start_as_current_span(self, *_, **__):
        return _NoOpSpan()
