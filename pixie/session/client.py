import asyncio
import inspect
from functools import wraps
import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, TypeVar, cast, overload
from uuid import uuid4

from langfuse import Langfuse, get_client
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

_langfuse = Langfuse()


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

    with _langfuse.start_as_current_observation(
        name="wait_for_input",
        as_type="tool",
    ):
        ret = await wait_for_input(req)
    await notify_server(
        SessionUpdate(
            session_id=exec_ctx.run_id,
            status="running",
            user_input=(
                ret.model_dump(mode="json") if isinstance(ret, BaseModel) else ret
            ),
        )
    )
    return ret


T_Func = TypeVar(
    "T_Func", bound=Callable[..., Awaitable[Any] | AsyncGenerator[Any, Any]]
)


def session(func: T_Func) -> T_Func:
    """Decorator to wrap functions with session context and instrumentation.

    Supports both async functions and async generators.

    Args:
        func: An async function or async generator function to wrap.

    Returns:
        A wrapped function that maintains the same signature and return type.
    """
    session_completed = False

    async def _session_setup(session_id: str):
        """Common setup logic for session."""
        ctx = execution_context.init_run(session_id)
        await connect_to_server(SESSION_RPC_SERVER_HOST, SESSION_RPC_PORT, session_id)

        update_queue = ctx.status_queue

        # handle updates from OTel, or any sources that are not part of user code
        async def notify_on_update():
            nonlocal session_completed
            while True:
                update = await update_queue.async_q.get()
                if not update:
                    await notify_server(
                        SessionUpdate(
                            session_id=session_id,
                            status="completed",
                        )
                    )
                    session_completed = True
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

        return task, langfuse

    async def _session_cleanup(
        session_id: str,
        task: asyncio.Task,
        langfuse,
        gen=None,
    ):
        """Common cleanup logic for session."""
        # Close generator if it was created
        if gen is not None:
            await gen.aclose()

        if langfuse:
            langfuse.flush()  # Ensure all spans are sent before disconnecting

        nonlocal session_completed
        if not session_completed:
            # Shield from cancellation to ensure "completed" is sent
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
                pass
            except Exception as e:
                logger.warning(f"Failed to send 'completed' status: {e}")

        task.cancel()
        disconnect_from_server()

    is_async_gen = inspect.isasyncgenfunction(func)
    if is_async_gen:

        @wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            session_id = uuid4().hex
            task, langfuse = await _session_setup(session_id)
            gen = None
            span = _langfuse.start_as_current_observation(
                name=func.__name__, as_type="chain"
            )
            try:
                span.__enter__()
                gen = func(*args, **kwargs)
                async for value in gen:
                    yield value
                span.__exit__(None, None, None)
            except Exception as e:
                span.__exit__(type(e), e, e.__traceback__)
                raise
            finally:
                await _session_cleanup(session_id, task, langfuse, gen)

        return async_gen_wrapper  # type: ignore
    else:

        @wraps(func)
        async def async_func_wrapper(*args, **kwargs):
            session_id = uuid4().hex
            task, langfuse = await _session_setup(session_id)
            span = _langfuse.start_as_current_observation(
                name=func.__name__, as_type="chain"
            )
            try:
                span.__enter__()
                result = await cast(Callable[..., Awaitable[Any]], func)(
                    *args, **kwargs
                )
                span.__exit__(None, None, None)
                return result
            except Exception as e:
                span.__exit__(type(e), e, e.__traceback__)
                raise
            finally:
                await _session_cleanup(session_id, task, langfuse)

        return async_func_wrapper  # type: ignore
