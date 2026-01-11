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
import docstring_parser

from pixie.types import PixieGenerator, UserInputRequirement
from pixie.utils import extract_schema_from_type


logger = logging.getLogger(__name__)


class RegistryItem:
    """Internal class to store registry item information.

    This class encapsulates all metadata and handlers for a registered application.
    """

    T = TypeVar("T", bound=BaseModel | JsonValue)

    def __init__(
        self,
        stream_handler: Callable[
            [JsonValue], AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]
        ],
        name: str,
        module: str,
        qualname: str,
        input_type: Optional[type[BaseModel]] | dict,
        user_input_type: Optional[type[BaseModel]] | dict | None,
        output_type: Optional[type[BaseModel]] | dict,
        short_description: Optional[str] = None,
        full_description: Optional[str] = None,
    ):
        """Initialize a RegistryItem.

        Args:
            stream_handler: The handler function for streaming.
            name: The function name.
            module: The module where the function is defined.
            qualname: The qualified name of the function.
            input_type: The expected input schema - either a Pydantic model type or JSON schema dict.
            user_input_type: The expected user input schema - Pydantic model type, JSON schema dict, or None.
            output_type: The expected output schema - either a Pydantic model type or JSON schema dict.
            short_description: A brief description of the application.
            full_description: A detailed description of the application.
        """
        self.stream_handler = stream_handler
        self.name = name
        self.module = module
        self.qualname = qualname
        self.input_type = input_type
        self.user_input_type = user_input_type
        self.output_type = output_type
        self.short_description = short_description
        self.full_description = full_description


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

    Args:
        type_hint: The type hint to analyze.

    Returns:
        A tuple of (yield_type, send_type) if both are Pydantic models or None.

    Raises:
        TypeError: If the provided type_hint is not an AsyncGenerator.
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


def _get_async_generator_hints(
    type_hint: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Extract both the yield and send type hints from an AsyncGenerator type hint.

    Args:
        type_hint: The type hint to analyze.

    Returns:
        A tuple of (yield_hint, send_hint) for any type hints.

    Raises:
        TypeError: If the provided type_hint is not an AsyncGenerator.
    """
    origin = get_origin(type_hint)
    if origin not in {AsyncGenerator, ABCAsyncGenerator}:
        raise TypeError("Provided type_hint is not an AsyncGenerator")

    args = get_args(type_hint)
    if len(args) != 2:
        return None, None

    return args[0], args[1]


def _is_async_generator_hint(type_hint: Any) -> bool:
    """Check if a type hint is an async generator.

    Args:
        type_hint: The type hint to check.

    Returns:
        True if the type hint is an async generator, False otherwise.
    """
    origin = get_origin(type_hint)
    return origin in {AsyncGenerator, ABCAsyncGenerator, AsyncIterable, AsyncIterator}


def _extract_input_type(func: Callable) -> Optional[type[BaseModel]]:
    """Extract input Pydantic model type from function signature.

    Returns only BaseModel types, used for automatic conversion.
    For schema extraction, use _extract_input_hint instead.
    """
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


def _extract_input_hint(func: Callable) -> Optional[Any]:
    """Extract input type hint from function signature.

    Returns any type hint, not just BaseModel types.
    Used for schema extraction.
    If function has no parameters, returns type(None) to indicate no input required.
    """
    try:
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            # No parameters: treat as None type (no input required)
            return type(None)

        param_name = params[0].name
        return hints.get(param_name)
    except (TypeError, ValueError, AttributeError):
        return None


def _value_to_json(
    value: JsonValue | BaseModel | UserInputRequirement,
) -> JsonValue | UserInputRequirement:
    """Convert a value to JSON-compatible format.

    Args:
        value: The value to convert.

    Returns:
        JSON-compatible value or UserInputRequirement.
    """
    if isinstance(value, UserInputRequirement):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    return value


def _json_to_value(
    json_data: JsonValue,
    model_type: Optional[type[BaseModel]],
) -> JsonValue | BaseModel:
    """Convert JSON data to a typed value.

    Args:
        json_data: The JSON data to convert.
        model_type: Optional Pydantic model type to convert to.

    Returns:
        The converted value, either as a BaseModel instance or raw JSON.
    """
    if model_type and isinstance(json_data, dict):
        return model_type(**json_data)
    return json_data


def _wrap_callable_handler(
    func: ApplicationCallable,
    input_type: Optional[type[BaseModel]],
) -> Callable[[JsonValue], AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]]:
    """Wrap a callable application handler to work with the streaming interface.

    Args:
        func: The application function to wrap.
        input_type: Optional Pydantic model type for input conversion.

    Returns:
        An async generator function that wraps the callable.
    """

    async def stream_handler(
        input_data: JsonValue,
    ) -> AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]:
        # Check if function accepts no parameters
        sig = inspect.signature(func)
        if not sig.parameters:
            # No-arg function: call without input
            result_or_awaitable = func()
        else:
            # Standard path: convert and pass input
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
    """Wrap a generator application handler to work with the streaming interface.

    Args:
        func: The generator application function to wrap.
        input_type: Optional Pydantic model type for input conversion.
        user_input_type: Optional Pydantic model type for user input conversion.

    Returns:
        An async generator function that wraps the generator.
    """

    async def stream_handler(
        input_data: JsonValue,
    ) -> AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]:
        # Check if function accepts no parameters
        sig = inspect.signature(func)
        if not sig.parameters:
            # No-arg generator: call without input
            generator_or_awaitable = func()
        else:
            # Standard path: convert and pass input
            processed_input: JsonValue | BaseModel = _json_to_value(
                input_data, input_type
            )
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


def _get_description_from_docstring(
    func: Callable,
) -> Tuple[str | None, str | None, docstring_parser.DocstringParam | None]:
    """Extract short, full description and input param from function docstring.

    Args:
        func: The function to extract docstring from.

    Returns:
        A tuple of (short_description, long_description, input_param).
    """
    doc = inspect.getdoc(func)
    if not doc:
        return None, None, None

    docstring = docstring_parser.parse(doc)
    if docstring.params:
        input_param = docstring.params[0]
    else:
        input_param = None

    return docstring.short_description, docstring.long_description, input_param


def _update_input_schema(
    input_schema: type | dict | None,
    input_param: docstring_parser.DocstringParam | None,
) -> type | dict | None:
    """Update input schema with information from docstring parameter.

    Args:
        input_schema: The schema to update.
        input_param: The docstring parameter containing description.

    Returns:
        The updated schema.
    """
    if isinstance(input_schema, dict) and input_param:
        input_schema["description"] = input_param.description
        input_schema["title"] = input_param.arg_name

    return input_schema


def _register_callable(
    func: ApplicationCallable,
    registry_key: str,
    name: str,
    module: str,
    qualname: str,
) -> None:
    """Register a callable application that returns a single result.

    Args:
        func: The application function to register.
        registry_key: Unique key for the registry.
        name: Function name.
        module: Module where the function is defined.
        qualname: Qualified name of the function.
    """
    input_model_type = _extract_input_type(func)
    input_hint = _extract_input_hint(func)
    # Only extract schema if there's an actual type hint
    input_schema = (
        extract_schema_from_type(input_hint) if input_hint is not None else None
    )

    output_schema = None
    try:
        hints = get_type_hints(func)
        output_hint = hints.get("return")
        # Only extract schema if there's an actual type hint
        output_schema = (
            extract_schema_from_type(output_hint) if output_hint is not None else None
        )
    except (TypeError, ValueError, AttributeError):
        pass

    stream_handler = _wrap_callable_handler(func, input_model_type)
    short_description, full_description, input_param = _get_description_from_docstring(
        func
    )
    input_schema = _update_input_schema(input_schema, input_param)

    _registry[registry_key] = RegistryItem(
        stream_handler=stream_handler,
        name=name,
        module=module,
        qualname=qualname,
        input_type=input_schema,
        user_input_type=None,
        output_type=output_schema,
        short_description=short_description,
        full_description=full_description,
    )


def _register_generator(
    func: ApplicationGenerator,
    registry_key: str,
    name: str,
    module: str,
    qualname: str,
) -> None:
    """Register a generator application that yields multiple results.

    Args:
        func: The generator application function to register.
        registry_key: Unique key for the registry.
        name: Function name.
        module: Module where the function is defined.
        qualname: Qualified name of the function.
    """
    input_model_type = _extract_input_type(func)
    input_hint = _extract_input_hint(func)
    # Only extract schema if there's an actual type hint
    input_schema = (
        extract_schema_from_type(input_hint) if input_hint is not None else None
    )

    output_model_type: Optional[type[BaseModel]] = None
    user_input_model_type: Optional[type[BaseModel]] = None
    output_hint: Optional[Any] = None
    user_input_hint: Optional[Any] = None

    try:
        hints = get_type_hints(func)
        return_hint = hints.get("return")
        # Extract Pydantic model types (for conversion)
        output_model_type, user_input_model_type = _get_async_generator_model_types(
            return_hint
        )
        # Extract any type hints (for schema generation)
        output_hint, user_input_hint = _get_async_generator_hints(return_hint)
    except (TypeError, ValueError, AttributeError):
        pass

    # For schema: prefer type hints, fall back to model types
    # Only extract schema if there's an actual type (not None from "no hint")
    if output_model_type is not None:
        # Pydantic model - extract_schema_from_type will return the class
        output_schema = extract_schema_from_type(output_model_type)
    elif output_hint is not None:
        # Other type hint
        output_schema = extract_schema_from_type(output_hint)
    else:
        output_schema = None

    # For user_input (send type), None means "no user input", not "null input"
    # So we keep it as None instead of converting to {"type": "null"}
    if user_input_model_type is not None:
        # Pydantic model
        user_input_schema = extract_schema_from_type(user_input_model_type)
    elif user_input_hint is not None and user_input_hint is not type(None):
        # Other type hint (but not None, which means no user input)
        user_input_schema = extract_schema_from_type(user_input_hint)
    else:
        user_input_schema = None

    stream_handler = _wrap_generator_handler(
        func, input_model_type, user_input_model_type
    )

    short_description, full_description, input_param = _get_description_from_docstring(
        func
    )
    input_schema = _update_input_schema(input_schema, input_param)

    _registry[registry_key] = RegistryItem(
        stream_handler=stream_handler,
        name=name,
        module=module,
        qualname=qualname,
        input_type=input_schema,
        user_input_type=user_input_schema,
        output_type=output_schema,
        short_description=short_description,
        full_description=full_description,
    )


# Overloads for common application patterns with specific type preservation

# When called with just func parameter (direct decoration: @pixie_app)


# Sync callable returning value (with parameter)
@overload
def pixie_app(func: Callable[[P], R]) -> Callable[[P], R]: ...


# Async callable returning value (with parameter)
@overload
def pixie_app(func: Callable[[P], Awaitable[R]]) -> Callable[[P], Awaitable[R]]: ...


# Async generator function (with parameter)
@overload
def pixie_app(
    func: Callable[[P], PixieGenerator[T, R]],
) -> Callable[[P], PixieGenerator[T, R]]: ...


# Async function returning async generator (with parameter)
@overload
def pixie_app(
    func: Callable[[P], Awaitable[PixieGenerator[T, R]]],
) -> Callable[[P], Awaitable[PixieGenerator[T, R]]]: ...


# Sync callable returning value (no parameters)
@overload
def pixie_app(func: Callable[[], R]) -> Callable[[], R]: ...


# Async callable returning value (no parameters)
@overload
def pixie_app(func: Callable[[], Awaitable[R]]) -> Callable[[], Awaitable[R]]: ...


# Async generator function (no parameters)
@overload
def pixie_app(
    func: Callable[[], PixieGenerator[T, R]],
) -> Callable[[], PixieGenerator[T, R]]: ...


# Async function returning async generator (no parameters)
@overload
def pixie_app(
    func: Callable[[], Awaitable[PixieGenerator[T, R]]],
) -> Callable[[], Awaitable[PixieGenerator[T, R]]]: ...


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

    # Extract metadata
    name = func.__name__
    module = func.__module__
    qualname = func.__qualname__

    registry_key = f"{module}.{qualname}"

    # Branch early based on function type
    is_generator = inspect.isasyncgenfunction(func)
    if not is_generator:
        try:
            hints = get_type_hints(func)
            is_generator = _is_async_generator_hint(hints.get("return"))
        except (TypeError, ValueError, AttributeError):
            pass

    if is_generator:
        _register_generator(
            cast(ApplicationGenerator, func), registry_key, name, module, qualname
        )
    else:
        _register_callable(
            cast(ApplicationCallable, func), registry_key, name, module, qualname
        )

    logger.info(
        "✅ Registered app: %s (%s.%s)",
        name,
        module,
        qualname,
    )

    return func


async def call_application(
    id: str,
    input_data: JsonValue,
) -> AsyncGenerator[UserInputRequirement | JsonValue, JsonValue]:
    """Call a registered application with automatic type conversion.

    Args:
        id: The application ID
        input_data: JSON-compatible input data

    Returns:
        Async generator streaming JSON-compatible output data
    """
    if id not in _registry:
        raise ValueError(f"Application '{id}' not found")

    app_info = _registry[id]
    generator = app_info.stream_handler(input_data)

    user_input: JsonValue | None = None
    try:
        while True:
            output = await generator.asend(user_input)
            user_input = yield output
    except StopAsyncIteration:
        return


def get_application(id: str) -> Optional[RegistryItem]:
    """Get a registered application info by id.

    Args:
        id: The application ID

    Returns:
        Registry entry or None if not found
    """
    return _registry.get(id)


def list_applications() -> list[str]:
    """List all registered application IDs.

    Returns:
        List of application IDs
    """
    return list(_registry.keys())


def clear_registry() -> None:
    """Clear all registered applications.

    This is primarily useful for testing to ensure a clean state.
    """
    _registry.clear()
