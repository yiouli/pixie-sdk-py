"""Application registry for managing registered AI applications."""

import logging
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
    Tuple,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
import inspect
from pydantic import BaseModel, JsonValue

from pixie.types import PixieGenerator, UserInputRequirement


logger = logging.getLogger(__name__)


class RegistryItem:
    """Internal class to store registry item information."""

    T = TypeVar("T", bound=BaseModel | JsonValue)

    def __init__(
        self,
        stream_handler: Callable[
            [JsonValue], AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]
        ],
        input_type: Optional[type[BaseModel]],
        user_input_type: Optional[type[BaseModel]],
        output_type: Optional[type[BaseModel]],
    ):
        """Initialize a RegistryItem.

        Args:
            stream_handler: The handler function for streaming.
            input_type (Optional[type[BaseModel]]): The expected input Pydantic model type.
            user_input_type (Optional[type[BaseModel]]): The expected user input Pydantic model type.
            output_type (Optional[type[BaseModel]]): The expected output Pydantic model type.
        """
        self.stream_handler = stream_handler
        self.input_type = input_type
        self.user_input_type = user_input_type
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
    PixieGenerator | Awaitable[PixieGenerator],
]
Application = ApplicationCallable | ApplicationGenerator

# TypeVars for decorator overloads
P = TypeVar("P", bound=BaseModel | JsonValue)  # Input parameter type
R = TypeVar("R", bound=BaseModel | JsonValue)  # Return type
T = TypeVar("T", bound=BaseModel | JsonValue)  # user Input type for


def _get_async_generator_model_types(
    type_hint: Any,
) -> Tuple[Optional[type[BaseModel]], Optional[type[BaseModel]]]:
    """Extract both the yield and send types from an AsyncGenerator type hint.

    Returns a tuple of (yield_type, send_type) if both are Pydantic models or None.
    Raises a TypeError if the provided type_hint is not an AsyncGenerator.
    """
    origin = get_origin(type_hint)
    if origin not in {AsyncGenerator, ABCAsyncGenerator}:
        raise TypeError("Provided type_hint is not an AsyncGenerator")

    args = get_args(type_hint)
    if len(args) != 2:
        return None, None

    yield_type, send_type = args
    yield_model = (
        yield_type
        if inspect.isclass(yield_type) and issubclass(yield_type, BaseModel)
        else None
    )
    send_model = (
        send_type
        if inspect.isclass(send_type) and issubclass(send_type, BaseModel)
        else None
    )

    return yield_model, send_model


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
        param_type = hints.get(param_name)
        if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
            return param_type
        return None
    except (TypeError, ValueError, AttributeError):
        return None


def _value_to_json(
    value: JsonValue | BaseModel | UserInputRequirement,
) -> JsonValue | UserInputRequirement:
    if isinstance(value, UserInputRequirement):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    return value


def _json_to_value(
    json_data: JsonValue,
    model_type: Optional[type[BaseModel]],
) -> JsonValue | BaseModel:
    if model_type and isinstance(json_data, dict):
        return model_type(**json_data)
    return json_data


def _wrap_callable_handler(
    func: ApplicationCallable,
    input_type: Optional[type[BaseModel]],
) -> Callable[[JsonValue], AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]]:
    async def stream_handler(
        input_data: JsonValue,
    ) -> AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]:
        result_or_awaitable = func(_json_to_value(input_data, input_type))

        if inspect.isawaitable(result_or_awaitable):
            result = await result_or_awaitable
        else:
            result = result_or_awaitable

        yield _value_to_json(result)

    return stream_handler


def _wrap_generator_handler(
    func: ApplicationGenerator,
    input_type: Optional[type[BaseModel]],
    user_input_type: Optional[type[BaseModel]],
) -> Callable[[JsonValue], AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]]:
    async def stream_handler(
        input_data: JsonValue,
    ) -> AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]:
        processed_input: JsonValue | BaseModel = _json_to_value(input_data, input_type)

        generator_or_awaitable = func(processed_input)

        # Handle both async generator functions and async functions returning generators
        if inspect.isawaitable(generator_or_awaitable):
            generator = await generator_or_awaitable
        else:
            generator = generator_or_awaitable

        try:
            user_input: JsonValue | BaseModel | None = None
            while True:
                result = await generator.asend(user_input)
                user_input_orig = yield _value_to_json(result)
                user_input = _json_to_value(user_input_orig, user_input_type)
        except StopAsyncIteration:
            return

    return stream_handler


def _register_callable(
    func: ApplicationCallable,
    registry_key: str,
) -> None:
    """Register a callable application that returns a single result."""
    input_type = _extract_input_type(func)

    output_type: Optional[type[BaseModel]] = None
    try:
        hints = get_type_hints(func)
        output_hint = hints.get("return")
        if inspect.isclass(output_hint) and issubclass(output_hint, BaseModel):
            output_type = output_hint

    except (TypeError, ValueError, AttributeError):
        output_type = None

    stream_handler = _wrap_callable_handler(func, input_type)

    _registry[registry_key] = RegistryItem(
        stream_handler=stream_handler,
        input_type=input_type,
        user_input_type=None,
        output_type=output_type,
    )


def _register_generator(
    func: ApplicationGenerator,
    registry_key: str,
) -> None:
    """Register a generator application that yields multiple results."""
    input_type = _extract_input_type(func)

    output_type: Optional[type[BaseModel]] = None
    user_input_type: Optional[type[BaseModel]] = None
    try:
        hints = get_type_hints(func)
        return_hint = hints.get("return")
        output_type, user_input_type = _get_async_generator_model_types(return_hint)
    except (TypeError, ValueError, AttributeError):
        pass

    stream_handler = _wrap_generator_handler(func, input_type, user_input_type)
    _registry[registry_key] = RegistryItem(
        stream_handler=stream_handler,
        input_type=input_type,
        user_input_type=user_input_type,
        output_type=output_type,
    )


# Overloads for common application patterns with specific type preservation

# When called with just func parameter (direct decoration: @pixie_app)


# Sync callable returning value
@overload
def pixie_app(func: Callable[[P], R]) -> Callable[[P], R]: ...


# Async callable returning value
@overload
def pixie_app(func: Callable[[P], Awaitable[R]]) -> Callable[[P], Awaitable[R]]: ...


# Async generator function
@overload
def pixie_app(
    func: Callable[[P], PixieGenerator[T, R]],
) -> Callable[[P], PixieGenerator[T, R]]: ...


# Async function returning async generator
@overload
def pixie_app(
    func: Callable[[P], Awaitable[PixieGenerator[T, R]]],
) -> Callable[[P], Awaitable[PixieGenerator[T, R]]]: ...


def pixie_app(func: Callable) -> Callable:
    """Register an application in the Pixie registry.

    This function can be used to register synchronous or asynchronous callables,
    as well as asynchronous generator functions. The registered application
    can then be invoked by name with automatic type conversion.

    Parameters
    ----------
    func : Callable
        The function to be registered.

    Returns:
    -------
    Callable
        The original function, unmodified.
    """

    registry_key = func.__name__

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
) -> AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]:
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
    generator = app_info.stream_handler(input_data)

    user_input: JsonValue | None = None
    try:
        while True:
            output = await generator.asend(user_input)
            user_input = yield output
    except StopAsyncIteration:
        return


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
