"""GraphQL schema for SDK server."""

import asyncio
import json
import logging
import uuid
from enum import Enum
from typing import AsyncGenerator, Callable, Coroutine, Optional, cast

from graphql import GraphQLError
from pydantic import JsonValue
import strawberry
import strawberry.experimental.pydantic
from strawberry.scalars import JSON

from langfuse import get_client
from pixie.registry import call_application, get_application
from pixie.types import (
    AppRunCancelled,
    BreakpointDetail as PydanticBreakpointDetail,
    AppRunUpdate as PydanticAppRunUpdate,
)
from pixie.otel_types import (
    OTLPKeyValue as PydanticOTLPKeyValue,
    OTLPSpanEvent as PydanticOTLPSpanEvent,
    OTLPSpanLink as PydanticOTLPSpanLink,
    OTLPStatus as PydanticOTLPStatus,
    OTLPSpan as PydanticOTLPSpan,
    OTLPInstrumentationScope as PydanticOTLPInstrumentationScope,
    OTLPScopeSpans as PydanticOTLPScopeSpans,
    OTLPResource as PydanticOTLPResource,
    OTLPResourceSpans as PydanticOTLPResourceSpans,
    OTLPTraceData as PydanticOTLPTraceData,
    PartialTraceData as PydanticPartialTraceData,
)

import pixie.execution_context as exec_ctx

logger = logging.getLogger(__name__)


# Create Strawberry enums matching Pydantic Literals
@strawberry.enum
class BreakpointTiming(Enum):
    """Mode for pausing execution."""

    BEFORE = "BEFORE"
    AFTER = "AFTER"


@strawberry.enum
class BreakpointType(Enum):
    """Types of pausible points in execution."""

    LLM = "LLM"
    TOOL = "TOOL"
    CUSTOM = "CUSTOM"


@strawberry.enum
class AppRunStatus(Enum):
    """Status of an application run."""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


@strawberry.enum
class TraceEventType(Enum):
    """Type of trace event.

    Indicates whether the trace data is from a span starting or other event.
    """

    SPAN_START = "span_start"


# Convert Pydantic models to Strawberry types with explicit field types
@strawberry.experimental.pydantic.type(model=PydanticBreakpointDetail)
class BreakpointDetail:
    """Represents the details of a breakpoint in execution.

    This class is a data model used to define the structure of a breakpoint,
    including its name, type, timing, and optional attributes.
    """

    span_name: strawberry.auto
    breakpoint_type: BreakpointType
    breakpoint_timing: BreakpointTiming
    span_attributes: JSON | None = None


# OTLP Trace Data Strawberry Types
@strawberry.experimental.pydantic.type(model=PydanticOTLPKeyValue)
class OTLPKeyValue:
    """OTLP key-value attribute."""

    key: strawberry.auto
    value: JSON  # Dict with string_value, int_value, etc.


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
    """Complete OTLP trace data structure.

    This matches the ExportTraceServiceRequest protobuf structure
    that Langfuse sends to its observability server.
    """


@strawberry.experimental.pydantic.type(model=PydanticPartialTraceData)
class PartialTraceData:
    """Partial trace data emitted at span start.

    Contains only the information available when a span begins,
    before it completes.
    """

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


@strawberry.type
class AppRunUpdate:
    """Represents updates for an application run.

    This class is used to define the structure of updates related to an application run,
    including the run ID, status, data, and optional breakpoint details.

    Attributes:
        run_id: The unique identifier of the application run.
        status: The current status of the application run.
        data: Additional data associated with the application run.
        breakpoint: Optional details about a breakpoint in the application run.
        trace: Optional trace data (either complete OTLP or partial).
    """

    run_id: strawberry.ID
    status: AppRunStatus
    data: Optional[str]
    breakpoint: Optional[BreakpointDetail]
    trace: Optional[TraceDataUnion]

    @classmethod
    def from_pydantic(cls, instance: PydanticAppRunUpdate):
        """Convert from Pydantic AppRunUpdate to Strawberry AppRunUpdate."""
        return cls(
            run_id=strawberry.ID(instance.run_id),
            status=AppRunStatus(instance.status),
            data=instance.data,
            breakpoint=BreakpointDetail.from_pydantic(instance.breakpoint)
            if instance.breakpoint
            else None,
            trace=_convert_trace_to_union(instance.trace),
        )


@strawberry.type
class Query:
    """GraphQL queries."""

    @strawberry.field
    async def health_check(self) -> str:
        """Health check endpoint."""
        logger.debug("Health check endpoint called")
        return "OK"


@strawberry.type
class Mutation:
    """GraphQL mutations."""

    @strawberry.mutation
    async def pause_run(
        self,
        run_id: str,
        timing: BreakpointTiming = BreakpointTiming.BEFORE,
        breakpoint_types: list[BreakpointType] | None = None,
    ) -> str:
        """Pause a run at a specific breakpoint.

        Args:
            run_id: The unique identifier of the run to pause.
            timing: The timing of the breakpoint (BEFORE or AFTER).
            breakpoint_types: A list of breakpoint types to apply, or None for all.

        Returns:
            The ID of the created breakpoint configuration.
        """
        try:
            breakpt_config = exec_ctx.set_breakpoint(
                run_id,
                timing.value,
                [t.value for t in (breakpoint_types or BreakpointType)],
            )
            return breakpt_config.id
        except ValueError as e:
            raise GraphQLError(str(e)) from e

    @strawberry.mutation
    async def resume_run(self, run_id: str) -> bool:
        """Resume a paused run.

        Args:
            run_id: The unique identifier of the run to resume.

        Returns:
            A boolean indicating whether the run was successfully resumed.
        """
        return exec_ctx.resume_run(run_id)


def _serialize_data(value: JsonValue | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _convert_trace_to_union(trace_dict: dict | None) -> Optional[TraceDataUnion]:
    """Convert trace data dict to TraceDataUnion.

    Args:
        trace_dict: Dict containing either OTLP trace data or partial trace data

    Returns:
        TraceDataUnion with the appropriate field set, or None
    """
    if trace_dict is None:
        return None

    # Check if it's a partial trace (has 'event' field)
    if trace_dict.get("event") == "span_start":
        partial = PydanticPartialTraceData(**trace_dict)
        return TraceDataUnion(
            partial_trace=PartialTraceData.from_pydantic(partial),
            otlp_trace=None,
        )

    # Otherwise it's a complete OTLP trace
    otlp = PydanticOTLPTraceData(**trace_dict)
    return TraceDataUnion(
        otlp_trace=OTLPTraceData.from_pydantic(otlp),
        partial_trace=None,
    )


def _create_app_run_in_thread(run: Coroutine) -> tuple[Coroutine, Callable[[], bool]]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run)

    return asyncio.to_thread(lambda: loop.run_until_complete(task)), task.cancel


@strawberry.type
class Subscription:
    """GraphQL subscriptions."""

    @strawberry.subscription
    async def run(
        self,
        name: str,
        input_data: JSON | None,
    ) -> AsyncGenerator[AppRunUpdate, None]:
        """Run an application and stream results.

        Args:
            name: The name of the registered application
            input_data: JSON string input data (or None)

        Yields:
            StatusUpdate: Status updates with results
        """
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        logger.info("Starting subscription for app=%s, run_id=%s", name, run_id)

        langfuse = get_client()

        if langfuse.auth_check():
            logger.info("Langfuse client initialized successfully.")
        else:
            logger.warning("Langfuse client authentication failed.")

        # Check if application exists
        if not get_application(name):
            logger.error("Application '%s' not found for run_id=%s", name, run_id)
            pydantic_update = PydanticAppRunUpdate(
                run_id=run_id,
                status="error",
                data=json.dumps({"error": f"Application '{name}' not found"}),
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)
            return

        task: Optional[asyncio.Task[None]] = None
        cancel: Optional[Callable[[], bool]] = None
        try:
            ctx = exec_ctx.init_run(run_id)
            status_queue = ctx.status_queue

            # Send initial running status
            pydantic_update = PydanticAppRunUpdate(
                run_id=run_id,
                status="running",
                data=json.dumps({"run_id": run_id}),
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)
            # set the context again after yield -- the fastapi/strawberry framework resets execution context
            exec_ctx.reload_run_context(run_id)

            # Create task for application execution
            async def run_application():
                try:
                    logger.info("Starting application execution for run_id=%s", run_id)
                    exec_ctx.reload_run_context(run_id)

                    async for item in call_application(
                        name,
                        cast(JsonValue, input_data),
                    ):
                        await exec_ctx.emit_status_update(
                            status="running", data=_serialize_data(item)
                        )

                    await exec_ctx.emit_status_update(status="completed")
                    logger.info("Application execution completed for run_id=%s", run_id)
                except (asyncio.CancelledError, AppRunCancelled):
                    logger.info("Application execution cancelled for run_id=%s", run_id)
                    await exec_ctx.emit_status_update(status="cancelled")
                    raise
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        "Application execution error for run_id=%s: %s", run_id, str(e)
                    )
                    await exec_ctx.emit_status_update(
                        status="error", data=json.dumps({"error": str(e)})
                    )
                    raise
                finally:
                    logger.debug(
                        "Application execution task ending for run_id=%s", run_id
                    )
                    langfuse.flush()
                    await exec_ctx.emit_status_update(status=None)

            run, cancel = _create_app_run_in_thread(run_application())
            task = asyncio.create_task(run)

            # Stream status updates from queue
            while True:
                # Wait for status update with timeout
                pydantic_update = await status_queue.get()
                if pydantic_update is None:
                    break

                yield AppRunUpdate.from_pydantic(pydantic_update)
                # set the context again after yield -- the fastapi/strawberry framework resets execution context
                exec_ctx.reload_run_context(run_id)

        finally:
            # Cancel the background task if it's still running
            if task is not None and not task.done():
                logger.info("Cancelling application task for run_id=%s", run_id)

                # Set cancellation flag to stop the child thread
                exec_ctx.cancel_run()

                # Cancel the child thread running the application
                if cancel:
                    cancel()

                # Cancel the asyncio task wrapper
                task.cancel()
            # Cleanup

            logger.debug("Unregistering run_id=%s", run_id)
            exec_ctx.unregister_run(run_id)


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
