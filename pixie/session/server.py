import asyncio
from collections.abc import Callable, Coroutine
import logging
import janus
from pydantic import JsonValue
from pixie.registry import app
from pixie.session.constants import SESSION_RPC_PORT
from pixie.session.rpc import (
    listen_to_client_connections,
    stop_server,
)
from pixie.session.types import SessionUpdate
from pixie.types import PixieGenerator


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


@app
async def run_session_server() -> PixieGenerator[SessionUpdate, dict[str, JsonValue]]:
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
            yield update
            # NOTE: below code have to be removed for background running app in web UX to work properly
            # this is only useful for testing when running the session server app in web UX
            # eanbling this would block all updates to web subscription after the InputRequired
            # until input is sent back from web UX
            # input_schema = update.user_input_schema
            # if input_schema is not None:
            #     i = yield InputRequired(
            #         dict,
            #         expected_schema={
            #             "type": "object",
            #             "properties": {"value": input_schema},
            #         },
            #     )
            #     try:
            #         await send_input_to_client(update.session_id, i["value"])
            #     except KeyError:
            #         # Client disconnected while we were waiting for UI input.
            #         # This is expected when user presses Ctrl+C.
            #         # Log and continue processing other clients.
            #         logger.warning(
            #             f"Client {update.session_id} disconnected before input "
            #             "could be sent. Continuing to serve other clients."
            #         )
    finally:
        cancel()
        task.cancel()
        # Properly close the janus queue
        update_queue.close()
        await update_queue.wait_closed()
