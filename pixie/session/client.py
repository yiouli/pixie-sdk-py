import asyncio
from functools import wraps
import logging
from typing import Any, Awaitable, Callable, TypeVar, overload
from uuid import uuid4

from langfuse import get_client
from pydantic import BaseModel, JsonValue

from pixie import execution_context
from pixie.server_utils import enable_instrumentations
from pixie.session.constants import SESSION_RPC_PORT, SESSION_RPC_SERVER_HOST
from pixie.session.rpc import (
    connect_to_server,
    disconnect_from_server,
    notify_server,
    wait_for_input,
)
from pixie.session.types import SessionUpdate
from pixie.types import InputRequired, InputType


logger = logging.getLogger(__name__)


async def print(data: str | JsonValue) -> None:
    exec_ctx = execution_context.get_current_context()
    if not exec_ctx:
        return

    await notify_server(
        SessionUpdate(
            session_id=exec_ctx.run_id,
            status="running",
            data=data,
        )
    )


T = TypeVar("T", bound=BaseModel)


@overload
async def input(
    prompt: str | None = None,
    *,
    expected_type: type[InputType],
) -> InputType: ...


@overload
async def input(
    prompt: str | None = None,
    *,
    expected_type: None = None,
) -> str: ...


async def input(
    prompt: str | None = None,
    *,
    expected_type: type[InputType] | None = None,
) -> InputType | str:
    exec_ctx = execution_context.get_current_context()
    if not exec_ctx:
        raise RuntimeError("No execution context found.")

    req = InputRequired(expected_type or str)
    update = SessionUpdate(
        session_id=exec_ctx.run_id,
        data=prompt,
        status="waiting",
        user_input_schema=req.get_json_schema(),
    )

    await notify_server(update)

    return await wait_for_input(req)


def session(func: Callable[..., Awaitable[Any]]):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        session_id = uuid4().hex
        ctx = execution_context.init_run(session_id)
        await connect_to_server(SESSION_RPC_SERVER_HOST, SESSION_RPC_PORT, session_id)

        update_queue = ctx.status_queue
        completed = False

        # handle updates from OTel, or any sources that are not part of user code
        async def notify_on_update():
            while True:
                update = await update_queue.async_q.get()
                if not update:
                    await notify_server(
                        SessionUpdate(
                            session_id=session_id,
                            status="completed",
                        )
                    )
                    nonlocal completed
                    completed = True
                    break
                await notify_server(
                    SessionUpdate(
                        session_id=session_id,
                        status=update.status,
                        user_input=update.user_input,
                        data=update.data,
                        trace=update.trace,
                        prompt_for_span=update.prompt_for_span,
                    )
                )

        task = asyncio.create_task(notify_on_update())
        langfuse = None
        try:
            enable_instrumentations()
            langfuse = get_client()

            if langfuse.auth_check():
                logger.debug("Langfuse client authenticated")
            else:
                logger.debug(
                    "Langfuse authentication failed, continuing in Pixie-only mode"
                )
            await notify_server(
                SessionUpdate(
                    session_id=session_id,
                    status="running",
                )
            )
            return await func(*args, **kwargs)
        finally:
            if langfuse:
                langfuse.flush()  # Ensure all spans are sent before disconnecting

            if not completed:
                # Shield from cancellation to ensure "completed" is sent
                # even when the task is cancelled (e.g., Ctrl+C)
                try:
                    await asyncio.shield(
                        notify_server(
                            SessionUpdate(
                                session_id=session_id,
                                status="completed",
                            )
                        )
                    )
                except asyncio.CancelledError:
                    # Re-raise to propagate cancellation after cleanup
                    pass
                except Exception as e:
                    # Log but don't fail on notification errors during cleanup
                    logger.warning(f"Failed to send 'completed' status: {e}")

            task.cancel()
            disconnect_from_server()

    return wrapper
