"""GraphQL schema for SDK server."""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Callable, Coroutine, Optional, cast

from importlib.metadata import PackageNotFoundError, version
from graphql import GraphQLError
import janus
from pydantic import BaseModel, JsonValue
import strawberry
from strawberry.scalars import JSON

from langfuse import get_client
import pixie.execution_context as exec_ctx
from pixie.agents.graphql import AgentMutation, AgentSubscription
from pixie.prompts.graphql import Mutation as PromptsMutation, Query as PromptsQuery
from pixie.registry import call_application, get_application, list_applications
from pixie.session.rpc import (
    get_connected_sessions,
    send_input_to_client,
    wait_for_client_update,
)
from pixie.storage.graphql import StorageMutation, StorageQuery
from pixie.types import (
    AppRunCancelled,
    AppRunUpdate as PydanticAppRunUpdate,
    InputRequired,
)
from pixie.strawberry_types import (
    AppInfo,
    AppRunUpdate,
    BreakpointTiming,
    BreakpointType,
    SessionInfo,
)

# Global registry for input queues per run
_input_queues: dict[str, janus.Queue] = {}


def _get_input_queue(run_id: str) -> janus.Queue:
    """Get or create the input queue for a given run ID.

    Args:
        run_id: The unique identifier of the run
    Returns:
        The queue for this run
    """
    if run_id not in _input_queues:
        _input_queues[run_id] = janus.Queue()
    return _input_queues[run_id]


logger = logging.getLogger(__name__)


@strawberry.type
class _Query:
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
    async def list_sessions(self) -> list[SessionInfo]:
        """List all active session IDs.

        Returns:
            A list of active session IDs.
        """
        sessions = await get_connected_sessions()
        return [SessionInfo.from_pydantic(s) for s in sessions]

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


@strawberry.type
class _Mutation:
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
            q.sync_q.put(input_data)
            logger.info("User input sent to run_id=%s", run_id)
            return True
        except Exception as e:
            raise GraphQLError(f"Failed to send input: {str(e)}") from e

    @strawberry.mutation
    async def send_session_input(self, session_id: str, input_data: JSON) -> bool:
        """Send user input to a running session.

        Args:
            session_id: The unique identifier of the session to send input to.
            input_data: The input data to send to the session.

        Returns:
            True if the input was successfully sent.

        Raises:
            GraphQLError: If sending input fails.
        """

        try:
            await send_input_to_client(session_id, cast(JsonValue, input_data))
            logger.info("User input sent to session_id=%s", session_id)
            return True
        except Exception as e:
            raise GraphQLError(f"Failed to send session input: {str(e)}") from e


def _create_app_run_in_thread(run: Coroutine) -> tuple[Coroutine, Callable[[], None]]:
    """Create a new event loop in a thread to run the application.

    Args:
        run: The coroutine to execute in the new event loop.

    Returns:
        A tuple of (thread_coroutine, cancel_function) where thread_coroutine
        is a coroutine that runs the loop and cancel_function can cancel the task
        in a thread-safe manner.
    """
    loop = asyncio.new_event_loop()
    task: asyncio.Task | None = None

    def run_loop():
        nonlocal task
        asyncio.set_event_loop(loop)
        task = loop.create_task(run)
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

    def cancel():
        """Thread-safe cancellation of the task."""
        if task is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)

    return asyncio.to_thread(run_loop), cancel


@strawberry.type
class _Subscription:
    """GraphQL subscriptions."""

    @strawberry.subscription
    async def run_session(self, session_id: str) -> AsyncGenerator[AppRunUpdate, None]:
        """Subscribe to application run updates for a session.

        Args:
            session_id: The unique identifier of the session.
        Yields:
            AppRunUpdate: Status updates including run status, data, breakpoints, and traces.
        """
        try:
            while True:
                session_update = await wait_for_client_update(session_id)
                app_update = PydanticAppRunUpdate(
                    run_id=session_update.session_id,
                    status=session_update.status,
                    time_unix_nano=session_update.time_unix_nano,
                    user_input_schema=session_update.user_input_schema,
                    user_input=session_update.user_input,
                    data=session_update.data,
                    trace=session_update.trace,
                    prompt_for_span=session_update.prompt_for_span,
                )
                yield AppRunUpdate.from_pydantic(app_update)
                if app_update.status in (
                    "completed",
                    "error",
                    "cancelled",
                ):
                    break
        except Exception as e:
            logger.error(
                "Error in run_session subscription for session_id=%s: %s",
                session_id,
                str(e),
            )
            raise GraphQLError(f"Subscription error: {str(e)}") from e

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
                time_unix_nano=str(time.time_ns()),
                data=json.dumps({"error": f"Application '{id}' not found"}),
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)
            return

        task: Optional[asyncio.Task[None]] = None
        cancel: Optional[Callable[[], None]] = None
        ctx = exec_ctx.init_run(run_id)
        try:
            status_queue = ctx.status_queue

            # Send initial running status
            pydantic_update = PydanticAppRunUpdate(
                run_id=run_id,
                status="running",
                time_unix_nano=str(time.time_ns()),
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)
            # set the context again after yield -- the fastapi/strawberry framework resets execution context
            exec_ctx.reload_run_context(run_id)

            # Create task for application execution
            async def run_application():
                app_gen = None
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
                                user_input = await input_q.async_q.get()
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
                    # Close the application generator to trigger its cleanup
                    if app_gen is not None:
                        await app_gen.aclose()
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


@strawberry.type
class Query(_Query, PromptsQuery, StorageQuery):
    """Combined GraphQL query schema."""


@strawberry.type
class Mutation(_Mutation, AgentMutation, PromptsMutation, StorageMutation):
    """Combined GraphQL mutation schema."""


@strawberry.type
class Subscription(_Subscription, AgentSubscription):
    """Combined GraphQL subscription schema."""


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
