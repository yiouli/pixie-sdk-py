"""Application registry for managing registered AI applications."""

import logging
from functools import partial
from collections.abc import (
    AsyncGenerator as ABCAsyncGenerator,
    AsyncIterable,
    AsyncIterator,
)
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Optional,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
import inspect
from pydantic import BaseModel, JsonValue

from . import execution_context

logger = logging.getLogger(__name__)


class RegistryItem:
    """Internal class to store registry item information."""

    T = TypeVar("T", bound=BaseModel | JsonValue)

    def __init__(
        self,
        stream_handler: Callable[[JsonValue], AsyncGenerator[JsonValue, None]],
        input_type: Optional[type[BaseModel]],
        output_type: Optional[type[BaseModel]],
    ):
        self.stream_handler = stream_handler
        self.input_type = input_type
        self.output_type = output_type


# Registry stores both the handler and its type information
_registry: Dict[str, RegistryItem] = {}


# Internal type aliases for implementation logic
ApplicationCallable = Callable[
    [JsonValue | BaseModel], BaseModel | JsonValue | Awaitable[BaseModel | JsonValue]
]
# Generator can be async generator function OR async function returning AsyncGenerator
# Both are consumed the same way: async for item in func(input)
ApplicationGenerator = Callable[
    [JsonValue | BaseModel],
    AsyncGenerator[BaseModel | JsonValue, None]
    | Awaitable[AsyncGenerator[BaseModel | JsonValue, None]],
]
Application = ApplicationCallable | ApplicationGenerator

# TypeVars for decorator overloads
P = TypeVar("P", bound=BaseModel | JsonValue)  # Input parameter type (covariant-like)
R = TypeVar("R", bound=BaseModel | JsonValue)  # Return type (covariant)


def _get_pydantic_model_class(type_hint: Any) -> Optional[type[BaseModel]]:
    """Check if a type hint is a Pydantic model and return the class."""
    try:
        if inspect.isclass(type_hint) and issubclass(type_hint, BaseModel):
            return type_hint

        origin = get_origin(type_hint)
        if origin in {AsyncGenerator, ABCAsyncGenerator, AsyncIterable, AsyncIterator}:
            args = get_args(type_hint)
            if args:
                return _get_pydantic_model_class(args[0])
    except TypeError:
        pass
    return None


def _is_async_generator_hint(type_hint: Any) -> bool:
    origin = get_origin(type_hint)
    return origin in {AsyncGenerator, ABCAsyncGenerator, AsyncIterable, AsyncIterator}


def _extract_input_type(func: Application) -> Optional[type[BaseModel]]:
    """Extract input Pydantic model type from function signature."""
    try:
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            return None

        param_name = params[0].name
        return _get_pydantic_model_class(hints.get(param_name))
    except (TypeError, ValueError, AttributeError):
        return None


def _normalize_value(value: JsonValue | BaseModel) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    return value


def _wrap_callable_handler(
    func: ApplicationCallable,
    input_type: Optional[type[BaseModel]],
) -> Callable[[JsonValue], AsyncGenerator[JsonValue, None]]:
    async def stream_handler(input_data: JsonValue) -> AsyncGenerator[JsonValue, None]:
        processed_input: JsonValue | BaseModel = input_data
        if input_type and isinstance(input_data, dict):
            processed_input = input_type(**input_data)

        result_or_awaitable = func(processed_input)

        if inspect.isawaitable(result_or_awaitable):
            result = await result_or_awaitable
        else:
            result = result_or_awaitable

        yield _normalize_value(result)

    return stream_handler


def _wrap_generator_handler(
    func: ApplicationGenerator,
    input_type: Optional[type[BaseModel]],
) -> Callable[[JsonValue], AsyncGenerator[JsonValue, None]]:
    async def stream_handler(input_data: JsonValue) -> AsyncGenerator[JsonValue, None]:
        processed_input: JsonValue | BaseModel = input_data
        if input_type and isinstance(input_data, dict):
            processed_input = input_type(**input_data)

        generator_or_awaitable = func(processed_input)

        # Handle both async generator functions and async functions returning generators
        if inspect.isawaitable(generator_or_awaitable):
            generator = await generator_or_awaitable
        else:
            generator = generator_or_awaitable

        async for result in generator:
            yield _normalize_value(result)

    return stream_handler


def _register_callable(
    func: ApplicationCallable,
    registry_key: str,
) -> None:
    """Register a callable application that returns a single result."""
    input_type = _extract_input_type(func)

    try:
        hints = get_type_hints(func)
        output_type = _get_pydantic_model_class(hints.get("return"))
    except (TypeError, ValueError, AttributeError):
        output_type = None

    stream_handler = _wrap_callable_handler(func, input_type)

    _registry[registry_key] = RegistryItem(
        stream_handler=stream_handler,
        input_type=input_type,
        output_type=output_type,
    )


def _register_generator(
    func: ApplicationGenerator,
    registry_key: str,
) -> None:
    """Register a generator application that yields multiple results."""
    input_type = _extract_input_type(func)

    try:
        hints = get_type_hints(func)
        return_hint = hints.get("return")
        # For AsyncGenerator[T, None], extract T as the output type
        output_type = _get_pydantic_model_class(return_hint)
    except (TypeError, ValueError, AttributeError):
        output_type = None

    stream_handler = _wrap_generator_handler(func, input_type)

    _registry[registry_key] = RegistryItem(
        stream_handler=stream_handler,
        input_type=input_type,
        output_type=output_type,
    )


# Overloads for common application patterns with specific type preservation


# Sync callable returning value
@overload
def pixie_app(
    func: Callable[[P], R],
    *,
    name: str | None = None,
) -> Callable[[P], R]: ...


# Async callable returning value
@overload
def pixie_app(
    func: Callable[[P], Awaitable[R]],
    *,
    name: str | None = None,
) -> Callable[[P], Awaitable[R]]: ...


# Async generator function
@overload
def pixie_app(
    func: Callable[[P], AsyncGenerator[R, None]],
    *,
    name: str | None = None,
) -> Callable[[P], AsyncGenerator[R, None]]: ...


# Async function returning async generator
@overload
def pixie_app(
    func: Callable[[P], Awaitable[AsyncGenerator[R, None]]],
    *,
    name: str | None = None,
) -> Callable[[P], Awaitable[AsyncGenerator[R, None]]]: ...


F = TypeVar(
    "F",
    bound=Callable[
        [Any],
        BaseModel
        | JsonValue
        | Awaitable[BaseModel | JsonValue]
        | AsyncGenerator[BaseModel | JsonValue, None]
        | Awaitable[AsyncGenerator[BaseModel | JsonValue, None]],
    ],
)


@overload
def pixie_app(
    func: None = None,
    *,
    name: str | None = None,
) -> Callable[[F], F]: ...


def pixie_app(
    func: F | None = None,
    *,
    name: str | None = None,
) -> F | Callable[[F], F]:
    if func is None:
        return cast(Callable[[F], F], partial(pixie_app, name=name))

    registry_key = name if name is not None else func.__name__

    if registry_key in _registry:
        raise ValueError(f"Application '{registry_key}' is already registered")

    # Branch early based on function type
    is_generator = inspect.isasyncgenfunction(func)
    if not is_generator:
        try:
            hints = get_type_hints(func)
            is_generator = _is_async_generator_hint(hints.get("return"))
        except (TypeError, ValueError, AttributeError):
            pass

    if is_generator:
        _register_generator(cast(ApplicationGenerator, func), registry_key)
    else:
        _register_callable(cast(ApplicationCallable, func), registry_key)

    logger.info("✅ Registered application: %s", registry_key)

    return func


async def call_application(
    name: str,
    input_data: JsonValue,
) -> AsyncGenerator[JsonValue, None]:
    """Call a registered application with automatic type conversion.

    Args:
        name: The application name
        input_data: JSON-compatible input data

    Returns:
        Async generator streaming JSON-compatible output data
    """
    if name not in _registry:
        raise ValueError(f"Application '{name}' not found")

    app_info = _registry[name]
    handler = app_info.stream_handler
    ctx_obj = execution_context.get_execution_context()
    logger.info("Execution context inside call application '%s': %s", name, ctx_obj)

    async for output in handler(input_data):
        yield output


def get_application(name: str) -> Optional[RegistryItem]:
    """Get a registered application info by name.

    Args:
        name: The application name

    Returns:
        Registry entry or None if not found
    """
    return _registry.get(name)


def list_applications() -> list[str]:
    """List all registered application names.

    Returns:
        List of application names
    """
    return list(_registry.keys())


def clear_registry() -> None:
    """Clear all registered applications (useful for testing)."""
    _registry.clear()
