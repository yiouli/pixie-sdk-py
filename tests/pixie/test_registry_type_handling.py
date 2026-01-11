"""Test registry handling of type hints, especially None vs no hint."""

from typing import Any, Optional, cast
from pydantic import BaseModel

from pixie.registry import _extract_input_hint, _registry
from pixie.types import PixieGenerator, UserInputRequirement


class TestModel(BaseModel):
    """Test Pydantic model."""

    value: str


def test_extract_input_hint_with_none_type():
    """Test that _extract_input_hint correctly extracts None type."""

    async def func_with_none(_: None) -> str:
        return "test"

    hint = _extract_input_hint(func_with_none)
    assert hint is type(None), f"Expected type(None), got {hint}"


def test_extract_input_hint_with_no_hint():
    """Test that _extract_input_hint returns None for no type hint."""

    async def func_no_hint(x):
        return "test"

    hint = _extract_input_hint(func_no_hint)
    assert hint is None, f"Expected None (no hint), got {hint}"


def test_extract_input_hint_with_string():
    """Test that _extract_input_hint correctly extracts str type."""

    async def func_with_str(query: str) -> str:
        return query

    hint = _extract_input_hint(func_with_str)
    assert hint is str, f"Expected str, got {hint}"


def test_extract_input_hint_with_pydantic_model():
    """Test that _extract_input_hint correctly extracts Pydantic model."""

    async def func_with_model(data: TestModel) -> str:
        return data.value

    hint = _extract_input_hint(func_with_model)
    assert hint is TestModel, f"Expected TestModel, got {hint}"


def test_register_callable_with_none_type():
    """Test that callable with None type gets correct schema."""
    from pixie import pixie_app

    @pixie_app
    async def test_app(_: None) -> str:
        return "test"

    # Check registry
    registry_key = f"{test_app.__module__}.{test_app.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be {"type": "null"}
    assert item.input_type == {"type": "null"}, (
        f"Expected null schema, got {item.input_type}"
    )


def test_register_callable_with_no_type():
    """Test that callable with no type hint gets None schema."""
    from pixie import pixie_app

    @pixie_app
    async def test_app_no_hint(x):
        return "test"

    # Check registry
    registry_key = f"{test_app_no_hint.__module__}.{test_app_no_hint.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be None (no hint)
    assert item.input_type is None, f"Expected None schema, got {item.input_type}"


def test_register_callable_with_string():
    """Test that callable with str type gets correct schema."""
    from pixie import pixie_app

    @pixie_app
    async def test_app_str(query: str) -> str:
        return query

    # Check registry
    registry_key = f"{test_app_str.__module__}.{test_app_str.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be {"type": "string"}
    assert item.input_type == {"type": "string"}, (
        f"Expected string schema, got {item.input_type}"
    )


def test_register_callable_with_pydantic_model():
    """Test that callable with Pydantic model returns model class."""
    from pixie import pixie_app

    @pixie_app
    async def test_app_model(data: TestModel) -> str:
        return data.value

    # Check registry
    registry_key = f"{test_app_model.__module__}.{test_app_model.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input type should be the model class itself
    assert item.input_type is TestModel, (
        f"Expected TestModel class, got {item.input_type}"
    )


def test_register_generator_with_none_type():
    """Test that generator with None type gets correct schema."""
    from pixie import pixie_app

    @pixie_app
    async def test_gen(_: None) -> PixieGenerator[str, str]:
        yield "test"
        user_input = yield ""  # Request user input
        yield f"Got: {user_input}"

    # Check registry
    registry_key = f"{test_gen.__module__}.{test_gen.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be {"type": "null"}
    assert item.input_type == {"type": "null"}, (
        f"Expected null schema, got {item.input_type}"
    )


def test_register_generator_output_types():
    """Test that generator output types are extracted correctly.

    Note: Due to how PixieGenerator type alias works, the type parameters
    don't map 1:1 to what gets extracted from AsyncGenerator.
    We're testing the actual behavior here.
    """
    from pixie import pixie_app

    @pixie_app
    async def test_gen_typed(query: str) -> PixieGenerator[str, int]:
        yield "Starting"
        count = yield UserInputRequirement(int)
        yield f"Count: {count}"

    # Check registry
    registry_key = f"{test_gen_typed.__module__}.{test_gen_typed.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be string
    assert item.input_type == {"type": "string"}, (
        f"Expected string schema, got {item.input_type}"
    )
    # Output (yield type) is a complex Union and not extracted
    assert item.output_type is None, f"Expected None output, got {item.output_type}"
    # User input (send type) is extracted from the AsyncGenerator send type
    assert item.user_input_type == {"type": "integer"}, (
        f"Expected integer user_input (send type), got {item.user_input_type}"
    )


def test_return_type_none():
    """Test that return type of None is handled correctly."""
    from pixie import pixie_app

    @pixie_app
    async def test_returns_none(x: str) -> None:
        print(x)

    # Check registry
    registry_key = f"{test_returns_none.__module__}.{test_returns_none.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Output schema should be {"type": "null"}
    assert item.output_type == {"type": "null"}, (
        f"Expected null output schema, got {item.output_type}"
    )


def test_optional_input_type():
    """Test that Optional[str] is handled correctly."""
    from pixie import pixie_app

    @pixie_app
    async def test_optional(query: Optional[str]) -> str:
        return query or "default"

    # Check registry
    registry_key = f"{test_optional.__module__}.{test_optional.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be anyOf with string and null
    assert isinstance(item.input_type, dict)
    schema = cast(dict[str, Any], item.input_type)
    assert "anyOf" in schema, f"Expected anyOf schema, got {item.input_type}"
    assert {"type": "string"} in schema["anyOf"]
    assert {"type": "null"} in schema["anyOf"]


def test_extract_input_hint_no_params():
    """Test that _extract_input_hint treats no-param functions as None type."""

    async def func_no_params() -> str:
        return "test"

    hint = _extract_input_hint(func_no_params)
    assert hint is type(None), f"Expected type(None) for no params, got {hint}"


def test_register_no_arg_callable():
    """Test that no-arg callable gets correct null schema."""
    from pixie import pixie_app

    @pixie_app
    async def test_no_args() -> str:
        return "no args needed"

    # Check registry
    registry_key = f"{test_no_args.__module__}.{test_no_args.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be {"type": "null"} for no-arg functions
    assert item.input_type == {"type": "null"}, (
        f"Expected null schema for no-arg function, got {item.input_type}"
    )


def test_register_no_arg_generator():
    """Test that no-arg generator gets correct null schema."""
    from pixie import pixie_app

    @pixie_app
    async def test_no_args_gen() -> PixieGenerator[str, None]:
        yield "output"

    # Check registry
    registry_key = f"{test_no_args_gen.__module__}.{test_no_args_gen.__qualname__}"
    assert registry_key in _registry

    item = _registry[registry_key]
    # Input schema should be {"type": "null"} for no-arg generators
    assert item.input_type == {"type": "null"}, (
        f"Expected null schema for no-arg generator, got {item.input_type}"
    )
