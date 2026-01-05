"""GraphQL schema for SDK server."""

import asyncio
import json
import logging
import queue
import uuid
from enum import Enum
from typing import AsyncGenerator, Callable, Coroutine, Optional, cast

from graphql import GraphQLError
from pydantic import BaseModel, JsonValue
import strawberry
import strawberry.experimental.pydantic
from strawberry.scalars import JSON

from langfuse import get_client
from pixie.registry import call_application, get_application, list_applications
from pixie.utils import get_json_schema_for_type
from pixie.types import (
    AppRunCancelled,
    BreakpointDetail as PydanticBreakpointDetail,
    AppRunUpdate as PydanticAppRunUpdate,
    UserInputRequirement,
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


# Global registry for input queues per run
_input_queues: dict[str, queue.Queue] = {}


def _get_input_queue(run_id: str) -> queue.Queue:
    """Get or create the input queue for a given run ID.

    Args:
        run_id: The unique identifier of the run
    Returns:
        The queue for this run
    """
    if run_id not in _input_queues:
        _input_queues[run_id] = queue.Queue()
    return _input_queues[run_id]


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
    WAITING = "waiting"
    CANCELLED = "cancelled"


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
    user_input_schema: Optional[JSON] = None
    user_input: Optional[JSON] = None
    data: Optional[JSON] = None
    breakpoint: Optional[BreakpointDetail] = None
    trace: Optional[TraceDataUnion] = None

    @classmethod
    def from_pydantic(cls, instance: PydanticAppRunUpdate):
        """Convert from Pydantic AppRunUpdate to Strawberry AppRunUpdate."""
        if instance.user_input_requirement:
            user_input_schema = JSON(
                get_json_schema_for_type(instance.user_input_requirement.expected_type)
            )
        else:
            user_input_schema = None
        return cls(
            run_id=strawberry.ID(instance.run_id),
            status=AppRunStatus(instance.status),
            user_input_schema=user_input_schema,
            user_input=JSON(instance.user_input),
            data=JSON(instance.data),
            breakpoint=BreakpointDetail.from_pydantic(instance.breakpoint)
            if instance.breakpoint
            else None,
            trace=_convert_trace_to_union(instance.trace),
        )


@strawberry.type
class AppInfo:
    """Schema information for a registered agent.

    Attributes:
        name: The unique name identifier for the agent.
        input_schema: JSON schema for the agent's input (either from Pydantic model or JsonValue type).
        user_input_schema: JSON schema for user input during execution (None for non-generator agents).
        output_schema: JSON schema for the agent's output (either from Pydantic model or JsonValue type).
    """

    name: str
    input_schema: Optional[JSON] = None
    user_input_schema: Optional[JSON] = None
    output_schema: Optional[JSON] = None


@strawberry.type
class Query:
    """GraphQL queries."""

    @strawberry.field
    async def health_check(self) -> str:
        """Health check endpoint."""
        logger.debug("Health check endpoint called")
        return "OK"

    @strawberry.field
    def list_apps(self) -> list[AppInfo]:
        """List all registered agents with their JSON schemas.

        Returns:
            A list of agent schemas containing name and input/output/user_input schemas.
        """
        agent_names = list_applications()
        agent_schemas = []

        for name in agent_names:
            app_info = get_application(name)
            if app_info is None:
                continue

            # Convert schema to JSON if it's a Pydantic model class
            def convert_to_json(schema_obj):
                if schema_obj is None:
                    return None
                if isinstance(schema_obj, type) and issubclass(schema_obj, BaseModel):
                    return JSON(schema_obj.model_json_schema())
                if isinstance(schema_obj, dict):
                    return JSON(schema_obj)
                return None

            agent_schemas.append(
                AppInfo(
                    name=name,
                    input_schema=convert_to_json(app_info.input_type),
                    user_input_schema=convert_to_json(app_info.user_input_type),
                    output_schema=convert_to_json(app_info.output_type),
                )
            )

        return agent_schemas


@strawberry.type
class Mutation:
    """GraphQL mutations."""

    @strawberry.mutation
    async def send_test(self, message: str) -> bool:
        """Send a test message to the test queue.

        Args:
            message: The test message to send.

        Returns:
            A boolean indicating whether the message was successfully sent.
        """
        try:
            q = _get_input_queue("test")
            q.put(message)
            logger.info("Test message sent: %s", message)
            return True
        except Exception as e:
            raise GraphQLError(f"Failed to send test message: {str(e)}") from e

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

    @strawberry.mutation
    async def send_input(self, run_id: str, input_data: JSON) -> bool:
        """Send user input to a running application.

        Args:
            run_id: The unique identifier of the run to send input to.
            input_data: The input data to send to the application.

        Returns:
            A boolean indicating whether the input was successfully sent.
        """

        try:
            q = _get_input_queue(run_id)
            q.put(input_data)
            logger.info("User input sent to run_id=%s", run_id)
            return True
        except Exception as e:
            raise GraphQLError(f"Failed to send input: {str(e)}") from e


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
    async def test(self) -> AsyncGenerator[str, None]:
        q = asyncio.Queue()

        async def run_app():
            while True:
                # Use asyncio to avoid blocking
                message = _get_input_queue("test").get()
                await q.put(message)

        run, cancel = _create_app_run_in_thread(run_app())
        task = asyncio.create_task(run)

        try:
            while True:
                message = await q.get()
                yield message
        finally:
            cancel()
            task.cancel()

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

                    app_gen = call_application(
                        name,
                        cast(JsonValue, input_data),
                    )

                    # Use asend pattern like in registry._wrap_generator_handler
                    user_input: JsonValue | None = None

                    try:
                        input_q = _get_input_queue(run_id)
                        while True:
                            item = await app_gen.asend(user_input)
                            # reset user input once it's sent to application
                            user_input = None

                            if isinstance(item, UserInputRequirement):
                                await exec_ctx.emit_status_update(
                                    status="waiting",
                                    user_input_requirement=item,
                                )
                                # Wait for input from queue
                                user_input = await asyncio.to_thread(input_q.get)
                                await exec_ctx.emit_status_update(
                                    status="running",
                                    user_input=user_input,
                                )
                            else:
                                await exec_ctx.emit_status_update(
                                    status="running",
                                    data=item,
                                )
                    except StopAsyncIteration:
                        pass

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
                        status="error", data={"error": str(e)}
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

            logger.debug("Unregistering run_id=%s", run_id)
            exec_ctx.unregister_run(run_id)

            # Clean up input queue
            if run_id in _input_queues:
                del _input_queues[run_id]
            exec_ctx.unregister_run(run_id)


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
