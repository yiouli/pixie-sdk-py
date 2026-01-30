import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
import logging
from pixie.session.constants import SESSION_RPC_PORT
from pixie.session.rpc import (
    listen_to_client_connections,
    stop_server,
)


logger = logging.getLogger(__name__)


def _run_in_thread(run: Coroutine) -> tuple[Coroutine, Callable[[], None]]:
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
        # First signal the server to stop via the running flag
        stop_server()
        # Then cancel the task in a thread-safe manner
        if task is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)

    return asyncio.to_thread(run_loop), cancel


@dataclass
class SessionServerRun:
    task: asyncio.Task
    cancel: Callable[[], None]


async def start_session_server() -> SessionServerRun:
    run, cancel = _run_in_thread(listen_to_client_connections(SESSION_RPC_PORT))
    task = asyncio.create_task(run)
    return SessionServerRun(task=task, cancel=cancel)
