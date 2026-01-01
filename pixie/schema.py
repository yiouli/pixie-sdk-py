"""GraphQL schema for SDK server."""

import asyncio
import json
import logging
import threading
import uuid
from enum import Enum
from typing import AsyncGenerator, Optional, cast

from pydantic import JsonValue
import strawberry
import strawberry.experimental.pydantic
from strawberry.scalars import JSON

from pixie.registry import call_application, get_application
from pixie.types import (
    PauseConfig,
    ExecutionContext,
    Breakpoint as PydanticBreakpoint,
    AppRunUpdate as PydanticAppRunUpdate,
    PauseResult as PydanticPauseResult,
    ResumeResult as PydanticResumeResult,
)
from pixie import execution_context as exec_ctx

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


# Convert Pydantic models to Strawberry types with explicit field types
@strawberry.experimental.pydantic.type(model=PydanticBreakpoint)
class Breakpoint:
    span_name: strawberry.auto
    breakpoint_type: BreakpointType
    breakpoint_timing: BreakpointTiming
    span_attributes: JSON | None = None


@strawberry.experimental.pydantic.type(model=PydanticAppRunUpdate)
class AppRunUpdate:
    run_id: strawberry.ID
    status: strawberry.auto
    data: strawberry.auto
    breakpoint: Optional[Breakpoint]


@strawberry.experimental.pydantic.type(model=PydanticPauseResult)
class PauseResult:
    success: strawberry.auto
    message: strawberry.auto
    run_id: strawberry.auto
    paused_state: JSON | None = None


@strawberry.experimental.pydantic.type(model=PydanticResumeResult)
class ResumeResult:
    success: strawberry.auto
    message: strawberry.auto
    run_id: strawberry.auto


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
        pause_mode: BreakpointTiming = BreakpointTiming.BEFORE,
        pausible_points: list[BreakpointType] | None = None,
    ) -> PauseResult:
        """Pause a running execution.

        Args:
            run_id: The ID of the run to pause
            pause_mode: Whether to pause before or after the next pausible point
            pausible_points: Types of pausible points to pause at

        Returns:
            PauseResult: Result of the pause operation
        """
        logger.info(
            "Pause request received for run_id=%s, mode=%s", run_id, pause_mode.value
        )

        # Check if run exists and is active
        if not exec_ctx.is_run_active(run_id):
            logger.warning(
                "Pause failed: run_id=%s not found or already completed", run_id
            )
            pydantic_result = PydanticPauseResult(
                success=False,
                message="Run not found or already completed",
                run_id=run_id,
            )
            return PauseResult.from_pydantic(pydantic_result)  # type: ignore[attr-defined]

        # Convert Strawberry enums to string literals for Pydantic
        pydantic_mode = pause_mode.value
        pydantic_points = [
            p.value
            for p in (
                pausible_points if pausible_points is not None else BreakpointType
            )
        ]

        # Create pause config
        pause_config = PauseConfig(
            mode=pydantic_mode,  # type: ignore[arg-type]
            pausible_points=pydantic_points,  # type: ignore[arg-type]
        )

        # Set pause config for the run
        success = exec_ctx.set_pause_config(run_id, pause_config)

        if not success:
            logger.error("Failed to set pause configuration for run_id=%s", run_id)
            pydantic_result = PydanticPauseResult(
                success=False,
                message="Failed to set pause configuration",
                run_id=run_id,
            )
            return PauseResult.from_pydantic(pydantic_result)  # type: ignore[attr-defined]

        # Emit pause_requested status
        await exec_ctx.emit_status_update(
            PydanticAppRunUpdate(
                run_id=run_id,
                status="pause_requested",
                data=json.dumps({"mode": pydantic_mode, "points": pydantic_points}),
            )
        )
        logger.info("Pause requested successfully for run_id=%s", run_id)

        pydantic_result = PydanticPauseResult(
            success=True,
            message="Pause requested successfully",
            run_id=run_id,
        )
        return PauseResult.from_pydantic(pydantic_result)  # type: ignore[attr-defined]

    @strawberry.mutation
    async def resume_run(self, run_id: str) -> ResumeResult:
        """Resume a paused execution.

        This mutation unblocks execution that was paused via pauseRun.
        The paused execution is blocking indefinitely until this mutation is called.

        Args:
            run_id: The ID of the run to resume

        Returns:
            ResumeResult: Result of the resume operation
        """
        logger.info("Resume request received for run_id=%s", run_id)

        # Check if run exists
        if not exec_ctx.is_run_active(run_id):
            logger.warning(
                "Resume failed: run_id=%s not found or already completed", run_id
            )
            pydantic_result = PydanticResumeResult(
                success=False,
                message="Run not found or already completed",
                run_id=run_id,
            )
            return ResumeResult.from_pydantic(pydantic_result)  # type: ignore[attr-defined]

        # Check if run is actually paused
        if not exec_ctx.is_run_paused(run_id):
            logger.warning("Resume failed: run_id=%s is not currently paused", run_id)
            pydantic_result = PydanticResumeResult(
                success=False,
                message="Cannot resume: run is not currently paused",
                run_id=run_id,
            )
            return ResumeResult.from_pydantic(pydantic_result)  # type: ignore[attr-defined]

        # Trigger resume
        success = exec_ctx.trigger_resume(run_id)

        if not success:
            logger.error("Failed to resume execution for run_id=%s", run_id)
            pydantic_result = PydanticResumeResult(
                success=False,
                message="Failed to resume execution",
                run_id=run_id,
            )
            return ResumeResult.from_pydantic(pydantic_result)  # type: ignore[attr-defined]

        logger.info("Execution resumed successfully for run_id=%s", run_id)
        return ResumeResult.from_pydantic(
            PydanticResumeResult(
                success=True,
                message="Execution resumed successfully",
                run_id=run_id,
            )
        )


def _serialize_data(value: JsonValue | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


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

        # Check if application exists
        if not get_application(name):
            logger.error("Application '%s' not found for run_id=%s", name, run_id)
            pydantic_update = PydanticAppRunUpdate(
                run_id=run_id,
                status="error",
                data=json.dumps({"error": f"Application '{name}' not found"}),
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)  # type: ignore[attr-defined]
            return

        # Create execution context with status queue and resume event
        status_queue: asyncio.Queue[PydanticAppRunUpdate] = asyncio.Queue()
        resume_event = asyncio.Event()
        resume_event.set()  # Start in resumed state

        sync_resume_event = threading.Event()
        sync_resume_event.set()  # Start in resumed state

        logger.debug("Registered run_id=%s and set execution context", run_id)

        # Start streaming status updates
        try:
            # Send initial running status
            pydantic_update = PydanticAppRunUpdate(
                run_id=run_id,
                status="running",
                data=json.dumps({"run_id": run_id}),
            )
            yield AppRunUpdate.from_pydantic(pydantic_update)  # type: ignore[attr-defined]

            # Register run and set context BEFORE creating the task
            ctx = ExecutionContext(
                run_id=run_id,
                status_queue=status_queue,
                resume_event=resume_event,
                sync_resume_event=sync_resume_event,
            )
            exec_ctx.register_run(ctx)
            exec_ctx.set_execution_context(ctx)

            # Create task for application execution
            async def run_application():
                try:
                    logger.debug("Starting application execution for run_id=%s", run_id)
                    # Context is already set, so it will be copied into this task
                    ctx = exec_ctx.get_execution_context()
                    logger.info(
                        "Execution context inside run_application for run_id=%s: %s",
                        run_id,
                        ctx,
                    )

                    async for item in call_application(
                        name,
                        cast(JsonValue, input_data),
                    ):
                        await status_queue.put(
                            PydanticAppRunUpdate(
                                run_id=run_id,
                                status="running",
                                data=_serialize_data(item),
                            )
                        )

                    await status_queue.put(
                        PydanticAppRunUpdate(
                            run_id=run_id,
                            status="completed",
                        )
                    )
                    logger.info("Application execution completed for run_id=%s", run_id)
                except (ValueError, TypeError, RuntimeError) as e:
                    logger.error(
                        "Application execution error for run_id=%s: %s", run_id, str(e)
                    )
                    await status_queue.put(
                        PydanticAppRunUpdate(
                            run_id=run_id,
                            status="error",
                            data=json.dumps({"error": str(e)}),
                        )
                    )

            # Start application execution
            app_task = asyncio.create_task(run_application())

            # Stream status updates from queue
            while True:
                # Wait for status update with timeout
                try:
                    pydantic_update = await asyncio.wait_for(
                        status_queue.get(), timeout=0.1
                    )
                    yield AppRunUpdate.from_pydantic(pydantic_update)  # type: ignore[attr-defined]

                    # Break on terminal status
                    if pydantic_update.status in ("completed", "error"):
                        break

                except asyncio.TimeoutError:
                    # Check if application task is done
                    if app_task.done():
                        # If task completed without putting terminal status, something went wrong
                        try:
                            await app_task  # Re-raise any exception
                        except Exception as e:  # Replace with specific exceptions
                            logger.error(
                                "Unexpected error in application task for run_id=%s: %s",
                                run_id,
                                str(e),
                            )
                            pydantic_update = PydanticAppRunUpdate(
                                run_id=run_id,
                                status="error",
                                data=json.dumps(
                                    {"error": f"Unexpected error: {str(e)}"}
                                ),
                            )
                            yield AppRunUpdate.from_pydantic(pydantic_update)  # type: ignore[attr-defined]
                            raise
                        break
                    # Continue waiting for updates
                    continue

        finally:
            # Cleanup
            logger.debug("Unregistering run_id=%s", run_id)
            exec_ctx.unregister_run(run_id)


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
