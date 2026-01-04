"""Pydantic models for OTLP (OpenTelemetry Protocol) trace data structures."""

from typing import Any, Literal, Optional
from pydantic import BaseModel


class OTLPKeyValue(BaseModel):
    """OTLP key-value attribute."""

    key: str
    value: Any  # Can be dict, string, int, bool - protobuf conversion varies


class OTLPSpanEvent(BaseModel):
    """OTLP span event."""

    name: str
    time_unix_nano: Optional[str] = None
    attributes: Optional[list[OTLPKeyValue]] = None
    dropped_attributes_count: Optional[int] = 0


class OTLPSpanLink(BaseModel):
    """OTLP span link."""

    trace_id: str
    span_id: str
    trace_state: Optional[str] = None
    attributes: Optional[list[OTLPKeyValue]] = None
    dropped_attributes_count: Optional[int] = 0
    flags: Optional[int] = 0


class OTLPStatus(BaseModel):
    """OTLP span status."""

    code: Optional[str] = None  # Can be '0', '1', '2' or 'STATUS_CODE_UNSET', etc.
    message: Optional[str] = None


class OTLPSpan(BaseModel):
    """OTLP span representation."""

    trace_id: str
    span_id: str
    trace_state: Optional[str] = None
    parent_span_id: Optional[str] = None
    name: str
    kind: Optional[str] = None  # Can be '0'-'5' or 'SPAN_KIND_INTERNAL', etc.
    start_time_unix_nano: Optional[str] = None
    end_time_unix_nano: Optional[str] = None
    attributes: Optional[list[OTLPKeyValue]] = None
    events: Optional[list[OTLPSpanEvent]] = None
    links: Optional[list[OTLPSpanLink]] = None
    status: Optional[OTLPStatus] = None
    dropped_attributes_count: Optional[int] = 0
    dropped_events_count: Optional[int] = 0
    dropped_links_count: Optional[int] = 0
    flags: Optional[int] = 0


class OTLPInstrumentationScope(BaseModel):
    """OTLP instrumentation scope (library info)."""

    name: str
    version: Optional[str] = None
    attributes: Optional[list[OTLPKeyValue]] = None
    dropped_attributes_count: Optional[int] = 0


class OTLPScopeSpans(BaseModel):
    """OTLP scope spans - groups spans by instrumentation scope."""

    scope: Optional[OTLPInstrumentationScope] = None
    spans: Optional[list[OTLPSpan]] = None
    schema_url: Optional[str] = None


class OTLPResource(BaseModel):
    """OTLP resource - describes the entity producing telemetry."""

    attributes: Optional[list[OTLPKeyValue]] = None
    dropped_attributes_count: Optional[int] = 0


class OTLPResourceSpans(BaseModel):
    """OTLP resource spans - groups spans by resource."""

    resource: Optional[OTLPResource] = None
    scope_spans: Optional[list[OTLPScopeSpans]] = None
    schema_url: Optional[str] = None


class OTLPTraceData(BaseModel):
    """Complete OTLP trace data structure.

    This matches the ExportTraceServiceRequest protobuf structure
    that Langfuse sends to its observability server.
    """

    resource_spans: Optional[list[OTLPResourceSpans]] = None


class PartialTraceData(BaseModel):
    """Partial trace data emitted at span start.

    Contains only the information available when a span begins,
    before it completes. This is different from the full OTLP
    trace structure.
    """

    event: Literal["span_start"] = "span_start"
    span_name: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    start_time_unix_nano: Optional[str] = None
    kind: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None


# Union type for trace data that can be either complete or partial
TraceData = OTLPTraceData | PartialTraceData
