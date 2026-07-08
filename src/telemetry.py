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
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Session telemetry context (Phase 5)
# ---------------------------------------------------------------------------

@dataclass
class TelemetryContext:
    """
    Per-session telemetry context propagated to every span.

    Built from :class:`src.session.SessionContext` and threaded through the
    agent phases so every LLM/tool span carries ``session.id`` (and optionally
    ``session.title`` / ``session.project``), making traces filterable per
    session in the Phoenix UI.
    """

    session_id: str
    title: Optional[str] = None
    project: Optional[str] = None

    def span_attributes(self) -> dict[str, str]:
        """Return the OpenTelemetry attributes that bind a span to this session."""
        attrs: dict[str, str] = {"session.id": self.session_id}
        if self.title:
            attrs["session.title"] = self.title
        if self.project:
            attrs["session.project"] = self.project
        return attrs


def telemetry_context_from_session(session) -> Optional[TelemetryContext]:
    """
    Build a :class:`TelemetryContext` from a :class:`SessionContext`.

    Returns None when the session has no ``phoenix_session_id`` (defensive —
    should not happen in normal flows since ``SessionManager`` defaults it).
    """
    phoenix_session_id = getattr(session, "phoenix_session_id", None)
    if not phoenix_session_id:
        return None
    return TelemetryContext(
        session_id=phoenix_session_id,
        title=getattr(session, "title", None),
        project=getattr(session, "phoenix_project", None),
    )


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


def span_llm_call(
    agent_role: str,
    milestone_id: str,
    model: str,
    session: Optional[TelemetryContext] = None,
):
    """
    Context manager wrapping an LLM call with a labelled span.

    Usage:
        with span_llm_call("worker", "M2", "local/qwen3-27b", session=tcx):
            result = call_llm(...)

    When ``session`` is provided, the span is tagged with ``session.id`` (and
    optionally ``session.title`` / ``session.project``) so traces are
    filterable per session in the Phoenix UI (Phase 5).
    """
    attributes = {
        "agent.role": agent_role,
        "mission.milestone_id": milestone_id,
        "llm.model": model,
    }
    if session is not None:
        attributes.update(session.span_attributes())
    return get_tracer().start_as_current_span(
        f"llm.{agent_role}",
        attributes=attributes,
    )


def span_tool_call(
    tool_name: str,
    milestone_id: str,
    session: Optional[TelemetryContext] = None,
):
    """Context manager wrapping a tool execution with a labelled span."""
    attributes = {
        "tool.name": tool_name,
        "mission.milestone_id": milestone_id,
    }
    if session is not None:
        attributes.update(session.span_attributes())
    return get_tracer().start_as_current_span(
        f"tool.{tool_name}",
        attributes=attributes,
    )


def span_mission_run(
    session_id: str,
    title: Optional[str] = None,
    project: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Context manager wrapping an entire ``MissionsRuntime.run()`` call.

    Creates a root ``mission.run`` span that parents all child LLM/tool spans
    for the session. Carries ``session.id`` so every descendant trace is
    filterable per session in Phoenix (Phase 5).
    """
    attributes: dict[str, str] = {"session.id": session_id}
    if title:
        attributes["session.title"] = title
    if project:
        attributes["session.project"] = project
    if model:
        attributes["mission.model"] = model
    return get_tracer().start_as_current_span(
        "mission.run",
        attributes=attributes,
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
