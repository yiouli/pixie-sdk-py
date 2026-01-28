import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, cast
import janus
from pydantic import BaseModel
from pixie.registry import app
from pixie.session.constants import SESSION_RPC_PORT
from pixie.session.rpc import (
    listen_to_client_connections,
    send_input_to_client,
    stop_server,
)
from pixie.session.types import SessionUpdate
from pixie.types import InputRequired, PixieGenerator


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


# emptry wrapper to avoid error on BaseModel.model_json_schema
class SessionInput(BaseModel):
    pass


@app
async def run_session_server() -> PixieGenerator[SessionUpdate, SessionInput]:
    update_queue = janus.Queue[SessionUpdate]()
    run, cancel = _run_in_thread(
        listen_to_client_connections(SESSION_RPC_PORT, update_queue)
    )
    task = asyncio.create_task(run)
    # Initial wait to allow server listening to start
    await asyncio.sleep(0.5)
    try:
        while True:
            # Use our local queue reference directly, not via _server_state
            # which is set in the child thread
            update = await update_queue.async_q.get()
            input_schema = update.user_input_schema
            # Clear the schema to avoid bloating the data sent to the client
            update.user_input_schema = None
            yield update
            if input_schema is not None:
                i = yield InputRequired(cast(dict[str, Any], input_schema))
                await send_input_to_client(update.session_id, i)
    finally:
        cancel()
        task.cancel()
        # Properly close the janus queue
        update_queue.close()
        await update_queue.wait_closed()
