"""Utility functions for JSON schema extraction and type handling."""

import typing
from typing import Any, Optional, Union
from pydantic import BaseModel


def is_allowed_type(type_hint: Any) -> bool:
    """Check if a type is one of the 3 supported categories.

    Supported:
    1. Simple JSON types: str, int, float, bool, None
    2. Complex JSON types: list[T], dict[str, T] where T is allowed
    3. Pydantic BaseModel types

    Also handles Optional[T] (Union[T, None]) where T is allowed.
    """
    if type_hint is None or type_hint is type(None):
        return True

    # Category 1: Simple types
    if type_hint in (str, int, float, bool):
        return True

    # Category 3: Pydantic model
    try:
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return True
    except (TypeError, ImportError):
        pass

    # Category 2: Complex types
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    # Handle Optional[T] (Union[T, None])
    if origin is typing.Union:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return is_allowed_type(non_none_types[0])
        return False  # Other unions not supported

    # Handle list[T]
    if origin is list:
        if args and len(args) == 1:
            return is_allowed_type(args[0])
        return True  # Untyped list is allowed

    # Handle dict[K, V] - K must be str
    if origin is dict:
        if args and len(args) == 2:
            if args[0] is not str:
                return False
            return is_allowed_type(args[1])
        return True  # Untyped dict is allowed

    # Untyped dict/list
    if type_hint is dict or type_hint is list:
        return True

    return False


def get_json_schema_for_type(expected_type: Any) -> dict | None:
    """Convert an allowed Python type to JSON schema.

    Args:
        expected_type: The type to convert to JSON schema

    Returns:
        A JSON schema dict representing the type, or None if not allowed
    """
    if not is_allowed_type(expected_type):
        return None

    # Handle None
    if expected_type is None or expected_type is type(None):
        return {"type": "null"}

    # Handle Pydantic model
    try:
        if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
            return expected_type.model_json_schema()
    except (TypeError, ImportError):
        pass

    # Handle simple types
    simple_type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    if expected_type in simple_type_mapping:
        return simple_type_mapping[expected_type]

    origin = typing.get_origin(expected_type)
    args = typing.get_args(expected_type)

    # Handle Optional[T] (Union[T, None])
    if origin is typing.Union:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            inner_schema = get_json_schema_for_type(non_none_types[0])
            if inner_schema:
                return {"anyOf": [inner_schema, {"type": "null"}]}

    # Handle list[T]
    if origin is list or expected_type is list:
        if args and len(args) == 1:
            item_schema = get_json_schema_for_type(args[0])
            return {"type": "array", "items": item_schema}
        return {"type": "array"}

    # Handle dict[str, T]
    if origin is dict or expected_type is dict:
        if args and len(args) == 2:
            value_schema = get_json_schema_for_type(args[1])
            return {"type": "object", "additionalProperties": value_schema}
        return {"type": "object"}

    return None


def extract_schema_from_type(
    type_hint: Any,
) -> Optional[Union[type[BaseModel], dict]]:
    """Extract schema from a type hint.

    For Pydantic models, returns the model class itself.
    For other allowed types (including None), returns a JSON schema dict.
    For unsupported types, returns None.

    Args:
        type_hint: The type hint to extract schema from.
                   Must be a valid type, not None for "no hint".

    Returns:
        Either a Pydantic model class, a JSON schema dict, or None
    """
    # Check if type is allowed
    if not is_allowed_type(type_hint):
        return None

    # If it's a Pydantic BaseModel, return the class itself
    try:
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint
    except (TypeError, ImportError):
        pass

    # For other allowed types, return JSON schema
    return get_json_schema_for_type(type_hint)
