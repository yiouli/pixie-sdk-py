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
from pixie.prompts.prompt import variables_definition_to_schema
from pixie.prompts.prompt_management import get_prompt, list_prompts
from pixie.registry import call_application, get_application, list_applications
from pixie.utils import get_json_schema_for_type
from pixie.types import (
    AppRunCancelled,
    BreakpointDetail as PydanticBreakpointDetail,
    AppRunUpdate as PydanticAppRunUpdate,
    InputRequired,
    PromptForSpan as PydanticPromptForSpan,
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
from importlib.metadata import PackageNotFoundError, version


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


@strawberry.experimental.pydantic.type(model=PydanticPromptForSpan)
class PromptForSpan:
    """Information about the prompt used in the application run.

    Attributes:
        trace_id: The trace ID associated with the prompt.
        span_id: The span ID associated with the prompt.
        prompt_id: Unique identifier of the prompt.
        version_id: Version identifier of the prompt.
        variables: Optional variables used in the prompt.
    """

    trace_id: strawberry.auto
    span_id: strawberry.auto
    prompt_id: strawberry.auto
    version_id: strawberry.auto
    variables: JSON | None = None


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
    prompt_for_span: Optional[PromptForSpan] = None

    @classmethod
    def from_pydantic(cls, instance: PydanticAppRunUpdate):
        """Convert from Pydantic AppRunUpdate to Strawberry AppRunUpdate.

        Args:
            instance: The Pydantic AppRunUpdate instance to convert.

        Returns:
            A Strawberry AppRunUpdate instance.
        """
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
    """Schema information for a registered agent.

    Attributes:
        id: The unique UUID identifier for the agent.
        name: The function name of the agent.
        qualified_name: The qualified name of the agent function.
        module: The module where the agent is defined.
        input_schema: JSON schema for the agent's input (either from Pydantic model or JsonValue type).
        user_input_schema: JSON schema for user input during execution (None for non-generator agents).
        output_schema: JSON schema for the agent's output (either from Pydantic model or JsonValue type).
    """

    id: str
    name: str
    qualified_name: str
    module: str
    short_description: Optional[str] = None
    full_description: Optional[str] = None
    input_schema: Optional[JSON] = None
    user_input_schema: Optional[JSON] = None
    output_schema: Optional[JSON] = None


@strawberry.type
class PromptMetadata:
    """Metadata for a registered prompt via create_prompt."""

    id: strawberry.ID
    variables_schema: JSON
    description: Optional[str] = None
    module: Optional[str] = None


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


@strawberry.type
class Prompt:
    """Full prompt information including versions."""

    id: strawberry.ID
    variables_schema: JSON
    versions: list[TKeyValue]
    default_version_id: str | None
    """default version id can only be None if versions is empty"""
    description: Optional[str] = None
    module: Optional[str] = None


@strawberry.type
class Query:
    """GraphQL queries."""

    @strawberry.field
    async def health_check(self) -> str:
        """Health check endpoint."""
        logger.debug("Health check endpoint called")
        try:
            version_str = version("pixie-sdk")
            logger.debug("Pixie SDK version: %s", version_str)
            return version_str
        except PackageNotFoundError as e:
            logger.warning("Failed to get Pixie SDK version: %s", str(e))
            return "0.0.0"

    @strawberry.field
    def list_apps(self) -> list[AppInfo]:
        """List all registered agents with their JSON schemas.

        Returns:
            A list of AppInfo objects containing id, name, qualified_name, module,
            and input/output/user_input schemas for each registered application.
        """

        agent_schemas = []

        for app_id in list_applications():
            # Convert schema to JSON if it's a Pydantic model class
            def convert_to_json(schema_obj):
                if schema_obj is None:
                    return None
                if isinstance(schema_obj, type) and issubclass(schema_obj, BaseModel):
                    return JSON(schema_obj.model_json_schema())
                if isinstance(schema_obj, dict):
                    return JSON(schema_obj)
                return None

            app_info = get_application(app_id)
            if app_info is None:
                continue

            agent_schemas.append(
                AppInfo(
                    id=app_id,
                    name=app_info.name,
                    qualified_name=app_info.qualname,
                    module=app_info.module,
                    short_description=app_info.short_description,
                    full_description=app_info.full_description,
                    input_schema=convert_to_json(app_info.input_type),
                    user_input_schema=convert_to_json(app_info.user_input_type),
                    output_schema=convert_to_json(app_info.output_type),
                )
            )

        return agent_schemas

    @strawberry.field
    def list_prompts(self) -> list[PromptMetadata]:
        """List all registered prompt templates.

        Returns:
            A list of PromptMetadata objects containing id, variables_schema,
            description, and module for each registered prompt.
        """
        return [
            PromptMetadata(
                id=strawberry.ID(p.prompt.id),
                variables_schema=JSON(
                    # NOTE: avoid p.get_variables_schema() to prevent potential fetching from storage
                    # this in theory could be different from the stored schema but in practice should not be
                    variables_definition_to_schema(p.prompt.variables_definition)
                ),
                description=p.description,
                module=p.module,
            )
            for p in list_prompts()
        ]

    @strawberry.field
    async def get_prompt(self, id: strawberry.ID) -> Prompt:
        """Get full prompt information including versions.

        Args:
            id: The unique identifier of the prompt.
        Returns:
            Prompt object containing id, variables_schema, versions,
            and default_version_id.
        Raises:
            GraphQLError: If prompt with given id is not found.
        """
        prompt_with_registration = get_prompt((str(id)))
        if prompt_with_registration is None:
            raise GraphQLError(f"Prompt with id '{id}' not found.")
        prompt = prompt_with_registration.prompt
        if not prompt.exists_in_storage():
            return Prompt(
                id=id,
                variables_schema=JSON(
                    # NOTE: avoid prompt.get_variables_schema() to prevent potential fetching from storage
                    variables_definition_to_schema(prompt.variables_definition)
                ),
                versions=[],
                default_version_id=None,
                description=prompt_with_registration.description,
                module=prompt_with_registration.module,
            )
        versions_dict = prompt.get_versions()
        versions = [TKeyValue(key=k, value=v) for k, v in versions_dict.items()]
        default_version_id: str = prompt.get_default_version_id()
        variables_schema = prompt.get_variables_schema()
        return Prompt(
            id=id,
            variables_schema=JSON(variables_schema),
            versions=versions,
            default_version_id=default_version_id,
            description=prompt_with_registration.description,
            module=prompt_with_registration.module,
        )


@strawberry.type
class Mutation:
    """GraphQL mutations."""

    @strawberry.mutation
    async def add_prompt_version(
        self,
        prompt_id: strawberry.ID,
        version_id: str,
        content: str,
        set_as_default: bool = False,
    ) -> str:
        """Add a new version to an existing prompt.

        Args:
            prompt_id: The unique identifier of the prompt.
            version_id: The identifier for the new version.
            content: The content of the new prompt version.
            set_as_default: Whether to set this version as the default.

        Returns:
            The updated BasePrompt object.
        """
        prompt_with_registration = get_prompt((str(prompt_id)))
        if prompt_with_registration is None:
            raise GraphQLError(f"Prompt with id '{prompt_id}' not found.")
        prompt = prompt_with_registration.prompt
        try:
            prompt.append_version(
                version_id=version_id,
                content=content,
                set_as_default=set_as_default,
            )
        except Exception as e:
            raise GraphQLError(f"Failed to add prompt version: {str(e)}") from e
        return "OK"

    @strawberry.mutation
    async def update_default_prompt_version(
        self,
        prompt_id: strawberry.ID,
        default_version_id: str,
    ) -> str:
        """Update the default version of an existing prompt.

        Args:
            prompt_id: The unique identifier of the prompt.
            default_version_id: The identifier of the version to set as default.

        Returns:
            True if the update was successful.
        """
        prompt_with_registration = get_prompt((str(prompt_id)))
        if prompt_with_registration is None:
            raise GraphQLError(f"Prompt with id '{prompt_id}' not found.")
        prompt = prompt_with_registration.prompt
        try:
            prompt.update_default_version_id(default_version_id)
        except Exception as e:
            raise GraphQLError(
                f"Failed to update default prompt version: {str(e)}"
            ) from e
        return "OK"

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
            True if the input was successfully sent.

        Raises:
            GraphQLError: If sending input fails.
        """

        try:
            q = _get_input_queue(run_id)
            q.put(input_data)
            logger.info("User input sent to run_id=%s", run_id)
            return True
        except Exception as e:
            raise GraphQLError(f"Failed to send input: {str(e)}") from e

    @strawberry.mutation
    async def run_llm(
        self,
        model: str,
        prompt_template: str,
        variables: JSON,
        tools: Optional[JSON] = None,
    ) -> str:
        """Run an LLM with the given prompt template and variables.

        Args:
            model: The model name to use (e.g., "openai:gpt-4").
            prompt_template: The prompt template text.
            variables: Variables to use in the prompt.
            tools: Optional tools configuration (not yet implemented).

        Returns:
            The LLM output as a string.

        Raises:
            GraphQLError: If the LLM call fails.
        """
        try:
            # Import pydantic_ai here to avoid circular imports
            from pydantic_ai import Agent

            # Create a simple agent with the given model
            agent = Agent(
                model, system_prompt=format(prompt_template, **cast(dict, variables))
            )

            # Run the agent with empty input since the prompt is in system_prompt
            # TODO: Support proper variable substitution in prompt template
            result = await agent.run()

            return result.output
        except Exception as e:
            logger.error("Error running LLM: %s", str(e))
            raise GraphQLError(f"Failed to run LLM: {str(e)}") from e


def _convert_trace_to_union(trace_dict: dict | None) -> Optional[TraceDataUnion]:
    """Convert trace data dict to TraceDataUnion.

    Args:
        trace_dict: Dict containing either OTLP trace data or partial trace data.

    Returns:
        TraceDataUnion with the appropriate field set, or None if trace_dict is None.
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
    """Create a new event loop in a thread to run the application.

    Args:
        run: The coroutine to execute in the new event loop.

    Returns:
        A tuple of (thread_coroutine, cancel_function) where thread_coroutine
        is a coroutine that runs the loop and cancel_function can cancel the task.
    """
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
        id: str,
        input_data: JSON | None,
    ) -> AsyncGenerator[AppRunUpdate, None]:
        """Run an application and stream results.

        Args:
            id: The UUID identifier of the registered application.
            input_data: JSON input data (or None if the application takes no input).

        Yields:
            AppRunUpdate: Status updates including run status, data, breakpoints, and traces.
        """
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        logger.info("Starting run: app=%s, run_id=%s", id, run_id)

        langfuse = get_client()

        if langfuse.auth_check():
            logger.debug("Langfuse client authenticated")
        else:
            logger.debug(
                "Langfuse authentication failed, continuing in Pixie-only mode"
            )

        # Check if application exists
        if not get_application(id):
            logger.error("Application not found: %s", id)
            pydantic_update = PydanticAppRunUpdate(
                run_id=run_id,
                status="error",
                data=json.dumps({"error": f"Application '{id}' not found"}),
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
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)
            # set the context again after yield -- the fastapi/strawberry framework resets execution context
            exec_ctx.reload_run_context(run_id)

            # Create task for application execution
            async def run_application():
                try:
                    logger.debug("Executing app run_id=%s", run_id)
                    exec_ctx.reload_run_context(run_id)

                    app_gen = call_application(
                        id,
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

                            if isinstance(item, InputRequired):
                                exec_ctx.emit_status_update(
                                    status="waiting",
                                    user_input_requirement=item,
                                )
                                # Wait for input from queue
                                user_input = await asyncio.to_thread(input_q.get)
                                exec_ctx.emit_status_update(
                                    status="running",
                                    user_input=user_input,
                                )
                            else:
                                exec_ctx.emit_status_update(
                                    status="running",
                                    data=item,
                                )
                    except StopAsyncIteration:
                        pass

                    exec_ctx.emit_status_update(status="completed")
                    logger.info("Completed run_id=%s", run_id)
                except (asyncio.CancelledError, AppRunCancelled):
                    logger.info("Cancelled run_id=%s", run_id)
                    exec_ctx.emit_status_update(status="cancelled")
                    raise
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Run error run_id=%s: %s", run_id, str(e))
                    exec_ctx.emit_status_update(status="error", data={"error": str(e)})
                    raise
                finally:
                    logger.debug("Ending task run_id=%s", run_id)
                    langfuse.flush()
                    exec_ctx.emit_status_update(status=None)

            run, cancel = _create_app_run_in_thread(run_application())
            task = asyncio.create_task(run)

            # Stream status updates from queue
            while True:
                # Wait for status update with timeout
                pydantic_update = await status_queue.async_q.get()
                if pydantic_update is None:
                    break

                yield AppRunUpdate.from_pydantic(pydantic_update)
                # set the context again after yield -- the fastapi/strawberry framework resets execution context
                exec_ctx.reload_run_context(run_id)

        finally:
            # Cancel the background task if it's still running
            if task is not None and not task.done():
                logger.debug("Cancelling task run_id=%s", run_id)

                # Set cancellation flag to stop the child thread
                exec_ctx.cancel_run()

                # Cancel the child thread running the application
                if cancel:
                    cancel()

                # Cancel the asyncio task wrapper
                task.cancel()

            logger.debug("Cleanup run_id=%s", run_id)

            # Close the janus queue to release resources
            if ctx and ctx.status_queue:
                ctx.status_queue.close()
                await ctx.status_queue.wait_closed()

            exec_ctx.unregister_run(run_id)

            # Clean up input queue
            if run_id in _input_queues:
                del _input_queues[run_id]


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
