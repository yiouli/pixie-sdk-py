"""Utility functions for JSON schema extraction and type handling."""

import typing
from typing import Any, Optional, Union
from pydantic import BaseModel, JsonValue


def get_json_schema_for_type(expected_type: Any) -> dict:
    """Convert a Python type to JSON schema.

    Args:
        expected_type: The type to convert to JSON schema

    Returns:
        A JSON schema dict representing the type
    """
    # Check if it's a Pydantic model
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
        type(None): {"type": "null"},
    }

    if expected_type in simple_type_mapping:
        return simple_type_mapping[expected_type]

    # Handle dict
    if expected_type is dict:
        return {"type": "object"}

    # Handle list
    if expected_type is list:
        return {"type": "array"}

    # Try to handle typing generics (Optional, List, Dict, etc.)
    origin = typing.get_origin(expected_type)

    if origin is dict:
        return {"type": "object"}
    elif origin is list:
        args = typing.get_args(expected_type)
        if args:
            # Recursively get schema for list item type
            item_schema = get_json_schema_for_type(args[0])
            return {"type": "array", "items": item_schema}
        return {"type": "array"}
    elif origin is typing.Union:
        # Handle Optional (Union[X, None])
        args = typing.get_args(expected_type)
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            # It's Optional[X]
            return get_json_schema_for_type(non_none_types[0])

    # Catch-all: completely loose schema (no type requirement)
    return {}


def extract_schema_from_type(
    type_hint: Optional[Any],
) -> Optional[Union[type[BaseModel], dict]]:
    """Extract schema from a type hint.

    For Pydantic models, returns the model class itself.
    For JsonValue, returns None (as it's the default JSON type).
    For other typed hints, returns a JSON schema dict.
    For None, returns None.

    Args:
        type_hint: The type hint to extract schema from

    Returns:
        Either a Pydantic model class, a JSON schema dict, or None
    """
    if type_hint is None:
        return None

    # If it's a Pydantic BaseModel, return the class itself
    try:
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint
    except (TypeError, ImportError):
        pass

    # Check if it's JsonValue - return None as it's the default/untyped case
    # JsonValue is typically Union[Dict[str, 'JsonValue'], List['JsonValue'], str, int, float, bool, None]
    if type_hint is JsonValue:
        return None

    origin = typing.get_origin(type_hint)

    # If it's a generic Union that looks like JsonValue, return None
    if origin is typing.Union:
        args = typing.get_args(type_hint)
        # Check if it's a complex union like JsonValue
        if len(args) > 3:  # JsonValue has many types
            return None

    # For other types with type information, return the JSON schema
    if origin is not None or type_hint in (str, int, float, bool, dict, list):
        return get_json_schema_for_type(type_hint)

    return None
