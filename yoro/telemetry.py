"""Small observability seam with an optional OpenTelemetry implementation."""
from __future__ import annotations

from contextlib import nullcontext


class Telemetry:
    def span(self, name: str, **attrs):
        return nullcontext()

    def event(self, name: str, **attrs) -> None:
        pass


class OpenTelemetry(Telemetry):
    def __init__(self, service_name: str = "yoro"):
        from opentelemetry import trace
        self.tracer = trace.get_tracer(service_name)

    def span(self, name: str, **attrs):
        return self.tracer.start_as_current_span(name, attributes=attrs)

    def event(self, name: str, **attrs) -> None:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(name, attrs)
