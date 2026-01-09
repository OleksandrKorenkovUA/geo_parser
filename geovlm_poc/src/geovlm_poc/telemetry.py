import logging
import os
import sys
from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - optional runtime dependency
    trace = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    OTLPSpanExporter = None


_logging_configured = False
_tracing_configured = False


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, *args, **kwargs):
        return None

    def record_exception(self, *args, **kwargs):
        return None

    def set_status(self, *args, **kwargs):
        return None


class _NoopTracer:
    def start_as_current_span(self, *args, **kwargs):
        return _NoopSpan()


def init_logging(level: Optional[str] = None) -> None:
    global _logging_configured
    if _logging_configured:
        return
    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    _logging_configured = True


def _select_exporter() -> Optional[object]:
    if OTLPSpanExporter is None or ConsoleSpanExporter is None:
        return None
    exporter = os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower()
    if not exporter or exporter == "none":
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            exporter = "otlp"
        elif os.environ.get("OTEL_CONSOLE_TRACES", "").strip().lower() in {"1", "true", "yes"}:
            exporter = "console"
        else:
            return None
    if exporter == "otlp":
        return OTLPSpanExporter()
    if exporter == "console":
        return ConsoleSpanExporter()
    return None


def init_tracing(service_name: str) -> None:
    global _tracing_configured
    if _tracing_configured:
        return
    if trace is None or Resource is None or TracerProvider is None:
        return
    exporter = _select_exporter()
    if exporter is None:
        _tracing_configured = True
        return
    service = os.environ.get("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create({"service.name": service})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracing_configured = True


def get_tracer(name: str):
    if trace is None:
        return _NoopTracer()
    return trace.get_tracer(name)
