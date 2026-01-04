"""Execution context management for pause/resume functionality."""

import asyncio
import logging
from contextvars import ContextVar
import threading
from typing import Dict, Optional, Sequence
from uuid import uuid4

from pixie.types import (
    AppRunCancelled,
    AppRunStatus,
    BreakpointDetail,
    BreakpointTiming,
    BreakpointType,
    ExecutionContext,
    BreakpointConfig,
    AppRunUpdate,
)

# ContextVar for storing execution context per async task
_execution_context: ContextVar[Optional[ExecutionContext]] = ContextVar(
    "execution_context", default=None
)

# Global registry of active runs for mutation access
_active_runs: Dict[str, ExecutionContext] = {}

# Logger
logger = logging.getLogger(__name__)


def init_run(run_id: str) -> ExecutionContext:
    """Initialize a new execution context for a run in the current context."""
    if _execution_context.get() is not None:
        raise RuntimeError("Execution context is already set")
    ctx = ExecutionContext(
        run_id=run_id,
        status_queue=asyncio.Queue(),
        resume_event=threading.Event(),
        breakpoint_config=None,
    )
    _execution_context.set(ctx)
    _active_runs[run_id] = ctx
    logger.info("Initialized execution context for run_id=%s", run_id)
    return ctx


def reload_run_context(run_id: str) -> None:
    """Reload the contextVar value from the global registry. this should be called whenever exec context changes."""
    ctx = get_run_context(run_id)
    if not ctx:
        raise ValueError(f"Run ID '{run_id}' not found")
    if ctx:
        _execution_context.set(ctx)


def get_current_breakpoint_config() -> Optional[BreakpointConfig]:
    """Get the current breakpoint configuration from execution context."""
    ctx = _execution_context.get()
    if ctx:
        return ctx.breakpoint_config
    return None


def get_current_context() -> Optional[ExecutionContext]:
    """Get the current execution context.

    Returns:
        The current ExecutionContext if available, None otherwise.
    """
    return _execution_context.get()


def unregister_run(run_id: str) -> None:
    """Unregister a run from the global active runs registry."""
    if run_id in _active_runs:
        del _active_runs[run_id]
        logger.info("Unregistered run: %s", run_id)


def get_run_context(run_id: str) -> Optional[ExecutionContext]:
    """Get execution context for a specific run ID."""
    return _active_runs.get(run_id)


async def emit_status_update(
    status: AppRunStatus | None,
    data: Optional[str] = None,
    breakpt: Optional[BreakpointDetail] = None,
    trace: Optional[dict] = None,
) -> None:
    """Emit a status update to the status queue if available."""
    ctx = _execution_context.get()
    if ctx:
        if status is None:
            update = None
            logger.debug("Emitted terminal status update for run %s", ctx.run_id)
        else:
            update = AppRunUpdate(
                run_id=ctx.run_id,
                status=status,
                data=data,
                breakpoint=breakpt,
                trace=trace,
            )
            logger.debug(
                "Emitted status update: %s for run %s", update.status, ctx.run_id
            )
        await ctx.status_queue.put(update)


def emit_status_update_sync(
    status: AppRunStatus | None,
    data: Optional[str] = None,
    breakpt: Optional[BreakpointDetail] = None,
    trace: Optional[dict] = None,
) -> None:
    """Emit a status update synchronously using put_nowait.

    This is a synchronous wrapper for emit_status_update that uses put_nowait
    instead of async put. Use this when you need to emit updates from
    synchronous code (e.g., span processors).

    Args:
        status: The status to emit, or None to end the stream
        data: Optional data string
        breakpt: Optional breakpoint details
        trace: Optional trace data dict
    """
    ctx = _execution_context.get()
    if ctx:
        if status is None:
            update = None
            logger.debug("Emitted terminal status update for run %s", ctx.run_id)
        else:
            update = AppRunUpdate(
                run_id=ctx.run_id,
                status=status,
                data=data,
                breakpoint=breakpt,
                trace=trace,
            )
            logger.debug(
                "Emitted status update: %s for run %s", update.status, ctx.run_id
            )
        ctx.status_queue.put_nowait(update)


def set_breakpoint(
    run_id: str,
    timing: BreakpointTiming,
    types: Sequence[BreakpointType],
) -> BreakpointConfig:
    """Set pause configuration for a run."""
    ctx = get_run_context(run_id)
    if not ctx:
        raise ValueError(f"Run ID '{run_id}' not found")

    breakpt_config = BreakpointConfig(
        id=uuid4().hex,
        timing=timing,
        breakpoint_types=list(types),
    )
    ctx.breakpoint_config = breakpt_config
    logger.info(
        "Breakpoint set for run_id=%s, timing=%s, types=%s",
        run_id,
        breakpt_config.timing,
        breakpt_config.breakpoint_types,
    )
    return breakpt_config


def wait_for_resume() -> None:
    """Block the current thread until the run is resumed."""
    ctx = _execution_context.get()
    if ctx is None:
        logger.warning(
            "No execution context found in current context, cannot wait for resume"
        )
        return

    logger.debug("Waiting for resume for run_id=%s, waiting for resume...", ctx.run_id)
    ctx.resume_event.wait()
    # Check one more time after waking up
    if ctx.cancelled:
        logger.info("Run cancelled during pause for run_id=%s", ctx.run_id)
        raise AppRunCancelled(f"Run {ctx.run_id} was cancelled")

    ctx.breakpoint_config = None
    logger.debug("Cleared pause config for run %s", ctx.run_id)
    ctx.resume_event.clear()
    logger.info("Resumed for run_id=%s", ctx.run_id)


def resume_run(run_id: str) -> bool:
    """Trigger resume for a paused run."""
    ctx = get_run_context(run_id)
    if not ctx:
        raise ValueError(f"Run ID '{run_id}' not found")

    if ctx.resume_event.is_set():
        # Already resumed or not paused
        logger.info(
            "Run_id=%s is not paused or already resumed",
            run_id,
        )
        return False

    ctx.resume_event.set()
    logger.debug("Trigger resume for run_id=%s", run_id)
    return True


def cancel_run() -> bool:
    """Cancel a running or paused run.

    Sets the cancellation flag which will cause the run to terminate.
    If the run is paused, it will also trigger the resume event to unblock it.

    Args:
        run_id: The ID of the run to cancel

    Returns:
        True if the run was found and cancelled, False otherwise
    """
    ctx = _execution_context.get()
    if not ctx:
        logger.warning(
            "No execution context found in current context, cannot cancel run"
        )
        return False

    if ctx.cancelled:
        logger.info("Run_id=%s is already cancelled", ctx.run_id)
        return False

    ctx.cancelled = True
    ctx.resume_event.set()
    logger.info("Resume run and set cancellation flag for run_id=%s", ctx.run_id)

    return True
