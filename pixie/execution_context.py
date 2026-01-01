"""Execution context management for pause/resume functionality."""

import asyncio
import os
import time
import logging
from contextvars import ContextVar
from typing import Dict, Optional

from pixie.types import ExecutionContext, PauseConfig, AppRunUpdate

# ContextVar for storing execution context per async task
execution_context: ContextVar[Optional[ExecutionContext]] = ContextVar(
    "execution_context", default=None
)

# Global registry of active runs for mutation access
active_runs: Dict[str, ExecutionContext] = {}

# Logger
logger = logging.getLogger(__name__)

# Timeout for abandoned paused runs (in seconds)
PAUSE_TIMEOUT_SECONDS = int(os.environ.get("PIXIE_PAUSE_TIMEOUT_SECONDS", "300"))


def get_execution_context() -> Optional[ExecutionContext]:
    """Get the current execution context from ContextVar."""
    return execution_context.get()


def set_execution_context(ctx: ExecutionContext) -> None:
    """Set the execution context in ContextVar."""
    execution_context.set(ctx)


def register_run(ctx: ExecutionContext) -> None:
    """Register a run in the global active runs registry."""
    active_runs[ctx.run_id] = ctx
    logger.info("Registered run: %s", ctx.run_id)


def unregister_run(run_id: str) -> None:
    """Unregister a run from the global active runs registry."""
    if run_id in active_runs:
        del active_runs[run_id]
        logger.info("Unregistered run: %s", run_id)


def get_run_context(run_id: str) -> Optional[ExecutionContext]:
    """Get execution context for a specific run ID."""
    return active_runs.get(run_id)


async def emit_status_update(update: AppRunUpdate) -> None:
    """Emit a status update to the status queue if available."""
    ctx = get_execution_context()
    if ctx and ctx.status_queue:
        await ctx.status_queue.put(update)
        logger.debug("Emitted status update: %s for run %s", update.status, ctx.run_id)


def set_pause_config(run_id: str, pause_config: PauseConfig) -> bool:
    """Set pause configuration for a run."""
    ctx = get_run_context(run_id)
    if not ctx:
        return False

    ctx.pause_config = pause_config
    logger.info(
        "Pause request for run_id=%s, mode=%s, points=%s",
        run_id,
        pause_config.mode,
        pause_config.pausible_points,
    )
    return True


def clear_pause_config(run_id: str) -> None:
    """Clear pause configuration for a run."""
    ctx = get_run_context(run_id)
    if ctx:
        ctx.pause_config = None
        logger.debug("Cleared pause config for run %s", run_id)


def trigger_resume(run_id: str) -> bool:
    """Trigger resume for a paused run."""
    ctx = get_run_context(run_id)
    if not ctx or not ctx.resume_event:
        return False

    if ctx.resume_event.is_set():
        # Already resumed or not paused
        return False

    # Calculate pause duration
    pause_duration_ms = None
    if ctx.pause_start_time:
        pause_duration_ms = int((time.time() - ctx.pause_start_time) * 1000)
        ctx.pause_start_time = None

    # Set both events
    ctx.resume_event.set()
    if ctx.sync_resume_event:
        ctx.sync_resume_event.set()

    logger.info(
        "Execution resumed for run_id=%s, paused_duration=%sms",
        run_id,
        pause_duration_ms,
    )

    # Schedule status update emission
    if ctx.status_queue:
        asyncio.create_task(
            ctx.status_queue.put(
                AppRunUpdate(
                    run_id=run_id,
                    status="resumed",
                    data=f'{{"paused_duration_ms": {pause_duration_ms}}}',
                )
            )
        )

    return True


def is_run_paused(run_id: str) -> bool:
    """Check if a run is currently paused."""
    ctx = get_run_context(run_id)
    if not ctx or not ctx.resume_event:
        return False
    return not ctx.resume_event.is_set()


def is_run_active(run_id: str) -> bool:
    """Check if a run is currently active."""
    return run_id in active_runs


async def cleanup_abandoned_runs() -> None:
    """Background task to cleanup abandoned paused runs."""
    logger.info(
        "Starting cleanup task for abandoned paused runs (timeout=%s)s",
        PAUSE_TIMEOUT_SECONDS,
    )

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            current_time = time.time()
            runs_to_cancel = []

            for run_id, ctx in active_runs.items():
                # Check if run is paused and has exceeded timeout
                if (
                    ctx.pause_start_time
                    and (current_time - ctx.pause_start_time) > PAUSE_TIMEOUT_SECONDS
                ):
                    runs_to_cancel.append(run_id)

            # Cancel abandoned runs
            for run_id in runs_to_cancel:
                ctx = active_runs.get(run_id)
                if ctx:
                    logger.warning(
                        "Auto-canceling run_id=%s after %s seconds of inactivity",
                        run_id,
                        PAUSE_TIMEOUT_SECONDS,
                    )

                    # Emit timeout status
                    if ctx.status_queue:
                        await ctx.status_queue.put(
                            AppRunUpdate(
                                run_id=run_id,
                                status="error",
                                data=f"Run cancelled due to pause timeout ({PAUSE_TIMEOUT_SECONDS}s)",
                            )
                        )

                    # Trigger resume to unblock any waiting code
                    if ctx.resume_event and not ctx.resume_event.is_set():
                        ctx.resume_event.set()

                    # Unregister run
                    unregister_run(run_id)

        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.error("Error in cleanup task: %s", str(e))


class CleanupTaskManager:
    """Manages the cleanup task lifecycle."""

    def __init__(self):
        self.cleanup_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the background cleanup task if not already running."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(cleanup_abandoned_runs())
            logger.info("Cleanup task started")

    def stop(self) -> None:
        """Stop the background cleanup task."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            logger.info("Cleanup task stopped")


# Instantiate the manager
cleanup_task_manager = CleanupTaskManager()


def start_cleanup_task() -> None:
    cleanup_task_manager.start()


def stop_cleanup_task() -> None:
    cleanup_task_manager.stop()
