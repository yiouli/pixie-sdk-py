"""Shared Strawberry types and enums for GraphQL schema.

This module contains all shared enums and types that are used across
multiple parts of the GraphQL schema to avoid circular imports.
"""

from enum import Enum
from typing import Optional

import strawberry
import strawberry.experimental.pydantic
from strawberry.scalars import JSON

from pixie.otel_types import (
    OTLPInstrumentationScope as PydanticOTLPInstrumentationScope,
    OTLPKeyValue as PydanticOTLPKeyValue,
    OTLPResource as PydanticOTLPResource,
    OTLPResourceSpans as PydanticOTLPResourceSpans,
    OTLPScopeSpans as PydanticOTLPScopeSpans,
    OTLPSpan as PydanticOTLPSpan,
    OTLPSpanEvent as PydanticOTLPSpanEvent,
    OTLPSpanLink as PydanticOTLPSpanLink,
    OTLPStatus as PydanticOTLPStatus,
    OTLPTraceData as PydanticOTLPTraceData,
    PartialTraceData as PydanticPartialTraceData,
)
from pixie.session.types import SessionInfo as PydanticSessionInfo
from pixie.storage.types import Message as PydanticMessage
from pixie.types import (
    AppRunUpdate as PydanticAppRunUpdate,
    BreakpointDetail as PydanticBreakpointDetail,
    PromptForSpan as PydanticPromptForSpan,
)


@strawberry.enum
class BreakpointTiming(str, Enum):
    """Mode for pausing execution."""

    BEFORE = "BEFORE"
    AFTER = "AFTER"


@strawberry.enum
class BreakpointType(str, Enum):
    """Types of pausible points in execution."""

    LLM = "LLM"
    TOOL = "TOOL"
    CUSTOM = "CUSTOM"


@strawberry.enum
class AppRunStatus(str, Enum):
    """Status of an application run."""

    running = "running"
    completed = "completed"
    error = "error"
    paused = "paused"
    waiting = "waiting"
    cancelled = "cancelled"
    unchanged = "unchanged"


@strawberry.enum
class Rating(str, Enum):
    """Rating for an LLM call or app run."""

    good = "good"
    bad = "bad"
    undecided = "undecided"


@strawberry.enum
class RatedBy(str, Enum):
    """Who made the rating."""

    user = "user"
    ai = "ai"
    system = "system"


@strawberry.enum
class MessageRole(str, Enum):
    """Role of a message in interaction logs."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    developer = "developer"


@strawberry.experimental.pydantic.input(model=PydanticMessage)
class MessageInput:
    """Message input for rating requests."""

    role: MessageRole
    content: JSON
    time_unix_nano: Optional[str]
    user_rating: Optional[Rating] = None
    user_feedback: Optional[str] = None


@strawberry.enum
class RunSource(str, Enum):
    """Source type for run records."""

    apps = "apps"
    sessions = "sessions"


@strawberry.enum
class TraceEventType(str, Enum):
    """Type of trace event.

    Indicates whether the trace data is from a span starting or other event.
    """

    SPAN_START = "span_start"


@strawberry.experimental.pydantic.type(model=PydanticBreakpointDetail)
class BreakpointDetail:
    """Represents the details of a breakpoint in execution."""

    span_name: strawberry.auto
    breakpoint_type: BreakpointType
    breakpoint_timing: BreakpointTiming
    span_attributes: JSON | None = None


@strawberry.experimental.pydantic.type(model=PydanticOTLPKeyValue)
class OTLPKeyValue:
    """OTLP key-value attribute."""

    key: strawberry.auto
    value: JSON


@strawberry.experimental.pydantic.type(model=PydanticOTLPStatus, all_fields=True)
class OTLPStatus:
    """OTLP span status."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPSpanEvent, all_fields=True)
class OTLPSpanEvent:
    """OTLP span event."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPSpanLink, all_fields=True)
class OTLPSpanLink:
    """OTLP span link."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPSpan, all_fields=True)
class OTLPSpan:
    """OTLP span representation."""


@strawberry.experimental.pydantic.type(
    model=PydanticOTLPInstrumentationScope, all_fields=True
)
class OTLPInstrumentationScope:
    """OTLP instrumentation scope (library info)."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPScopeSpans, all_fields=True)
class OTLPScopeSpans:
    """OTLP scope spans - groups spans by instrumentation scope."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPResource, all_fields=True)
class OTLPResource:
    """OTLP resource - describes the entity producing telemetry."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPResourceSpans, all_fields=True)
class OTLPResourceSpans:
    """OTLP resource spans - groups spans by resource."""


@strawberry.experimental.pydantic.type(model=PydanticOTLPTraceData, all_fields=True)
class OTLPTraceData:
    """Complete OTLP trace data structure."""


@strawberry.experimental.pydantic.type(model=PydanticPartialTraceData)
class PartialTraceData:
    """Partial trace data emitted at span start."""

    event: TraceEventType
    span_name: strawberry.auto
    trace_id: strawberry.auto
    span_id: strawberry.auto
    parent_span_id: strawberry.auto
    start_time_unix_nano: strawberry.auto
    kind: strawberry.auto
    attributes: JSON | None = None


@strawberry.type
class TraceDataUnion:
    """Union type for trace data - either complete OTLP or partial."""

    otlp_trace: Optional[OTLPTraceData] = None
    partial_trace: Optional[PartialTraceData] = None


@strawberry.experimental.pydantic.type(model=PydanticPromptForSpan)
class PromptForSpan:
    """Information about the prompt used in the application run."""

    trace_id: strawberry.auto
    span_id: strawberry.auto
    prompt_id: strawberry.auto
    version_id: strawberry.auto
    variables: JSON | None = None


@strawberry.type
class AppRunUpdate:
    """Represents updates for an application run."""

    run_id: strawberry.ID
    status: AppRunStatus
    time_unix_nano: str
    user_input_schema: Optional[JSON] = None
    user_input: Optional[JSON] = None
    data: Optional[JSON] = None
    breakpoint: Optional[BreakpointDetail] = None
    trace: Optional[TraceDataUnion] = None
    prompt_for_span: Optional[PromptForSpan] = None

    @classmethod
    def from_pydantic(cls, instance: PydanticAppRunUpdate):
        """Convert from Pydantic AppRunUpdate to Strawberry AppRunUpdate."""

        return cls(
            run_id=strawberry.ID(instance.run_id),
            status=AppRunStatus(instance.status),
            time_unix_nano=instance.time_unix_nano,
            user_input_schema=JSON(instance.user_input_schema),
            user_input=JSON(instance.user_input),
            data=JSON(instance.data),
            breakpoint=(
                BreakpointDetail.from_pydantic(instance.breakpoint)
                if instance.breakpoint
                else None
            ),
            trace=_convert_trace_to_union(instance.trace),
            prompt_for_span=(
                PromptForSpan.from_pydantic(instance.prompt_for_span)
                if instance.prompt_for_span
                else None
            ),
        )


@strawberry.type
class AppInfo:
    """Schema information for a registered agent."""

    id: str
    name: str
    qualified_name: str
    module: str
    short_description: Optional[str] = None
    full_description: Optional[str] = None
    input_schema: Optional[JSON] = None
    user_input_schema: Optional[JSON] = None
    output_schema: Optional[JSON] = None


@strawberry.input
class IKeyValue:
    """Key-value attribute."""

    key: str
    value: str


@strawberry.type
class TKeyValue:
    """Key-value attribute."""

    key: str
    value: str


@strawberry.experimental.pydantic.type(model=PydanticSessionInfo, all_fields=True)
class SessionInfo:
    """Session information for active sessions."""

    pass


def _convert_trace_to_union(trace_dict: dict | None) -> Optional[TraceDataUnion]:
    """Convert trace data dict to TraceDataUnion."""

    if trace_dict is None:
        return None

    if trace_dict.get("event") == TraceEventType.SPAN_START:
        partial = PydanticPartialTraceData(**trace_dict)
        return TraceDataUnion(
            partial_trace=PartialTraceData.from_pydantic(partial),
            otlp_trace=None,
        )

    otlp = PydanticOTLPTraceData(**trace_dict)
    return TraceDataUnion(
        otlp_trace=OTLPTraceData.from_pydantic(otlp),
        partial_trace=None,
    )
