import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, cast
import janus
from pydantic import BaseModel
from pixie.registry import app
from pixie.session.constants import SESSION_RPC_PORTS
from pixie.session.rpc import (
    listen_to_client_connections,
    send_input_to_client,
    wait_for_client_update,
)
from pixie.session.types import SessionUpdate
from pixie.types import InputRequired, PixieGenerator


def _run_in_thread(run: Coroutine) -> tuple[Coroutine, Callable[[], bool]]:
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


# emptry wrapper to avoid error on BaseModel.model_json_schema
class SessionInput(BaseModel):
    pass


@app
async def run_session_server() -> PixieGenerator[SessionUpdate, SessionInput]:
    update_queue = janus.Queue[SessionUpdate]()
    run, cancel = _run_in_thread(
        listen_to_client_connections(SESSION_RPC_PORTS, update_queue)
    )
    task = asyncio.create_task(run)
    # Initial wait to allow server listening to start
    await asyncio.sleep(0.5)
    try:
        while True:
            update = await wait_for_client_update()
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
