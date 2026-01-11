"""Test suite for pixie.utils module - JSON schema extraction and type handling."""

from typing import Optional
from pydantic import BaseModel

from pixie.utils import (
    is_allowed_type,
    get_json_schema_for_type,
    extract_schema_from_type,
)


# Test fixtures - Pydantic models
class SimpleModel(BaseModel):
    """Simple Pydantic model for testing."""

    name: str
    age: int


class NestedModel(BaseModel):
    """Nested Pydantic model for testing."""

    user: SimpleModel
    tags: list[str]


class TestIsAllowedType:
    """Test suite for is_allowed_type function."""

    def test_none_type(self):
        """Test that None type is allowed."""
        assert is_allowed_type(None) is True
        assert is_allowed_type(type(None)) is True

    def test_simple_types(self):
        """Test that simple JSON types are allowed."""
        assert is_allowed_type(str) is True
        assert is_allowed_type(int) is True
        assert is_allowed_type(float) is True
        assert is_allowed_type(bool) is True

    def test_pydantic_models(self):
        """Test that Pydantic BaseModel types are allowed."""
        assert is_allowed_type(SimpleModel) is True
        assert is_allowed_type(NestedModel) is True
        assert is_allowed_type(BaseModel) is True

    def test_untyped_collections(self):
        """Test that untyped dict and list are allowed."""
        assert is_allowed_type(dict) is True
        assert is_allowed_type(list) is True

    def test_typed_list_with_allowed_types(self):
        """Test that list[T] is allowed when T is allowed."""
        assert is_allowed_type(list[str]) is True
        assert is_allowed_type(list[int]) is True
        assert is_allowed_type(list[SimpleModel]) is True
        assert is_allowed_type(list[dict]) is True
        assert is_allowed_type(list[list[str]]) is True

    def test_typed_dict_with_str_key(self):
        """Test that dict[str, T] is allowed when T is allowed."""
        assert is_allowed_type(dict[str, str]) is True
        assert is_allowed_type(dict[str, int]) is True
        assert is_allowed_type(dict[str, SimpleModel]) is True
        assert is_allowed_type(dict[str, list[str]]) is True
        assert is_allowed_type(dict[str, dict[str, int]]) is True

    def test_typed_dict_with_non_str_key(self):
        """Test that dict[K, V] is NOT allowed when K is not str."""
        assert is_allowed_type(dict[int, str]) is False
        assert is_allowed_type(dict[bool, str]) is False
        assert is_allowed_type(dict[SimpleModel, str]) is False

    def test_optional_types(self):
        """Test that Optional[T] is allowed when T is allowed."""
        assert is_allowed_type(Optional[str]) is True
        assert is_allowed_type(Optional[int]) is True
        assert is_allowed_type(Optional[SimpleModel]) is True
        assert is_allowed_type(Optional[list[str]]) is True
        assert is_allowed_type(Optional[dict[str, int]]) is True

    def test_complex_nested_types(self):
        """Test complex nested allowed types."""
        # list of dicts with string values
        assert is_allowed_type(list[dict[str, str]]) is True
        # dict with list values
        assert is_allowed_type(dict[str, list[int]]) is True
        # Optional list of models
        assert is_allowed_type(Optional[list[SimpleModel]]) is True
        # dict with optional values
        assert is_allowed_type(dict[str, Optional[str]]) is True

    def test_unsupported_types(self):
        """Test that unsupported types return False."""
        # Tuple is not supported
        assert is_allowed_type(tuple) is False
        assert is_allowed_type(tuple[str, int]) is False
        # Set is not supported
        assert is_allowed_type(set) is False
        assert is_allowed_type(set[str]) is False
        # Complex unions (not Optional) are not supported
        from typing import Union

        assert is_allowed_type(Union[str, int, float]) is False

        # Custom classes (non-Pydantic)
        class CustomClass:
            pass

        assert is_allowed_type(CustomClass) is False

    def test_recursive_validation(self):
        """Test that recursive validation works for nested types."""
        # list[unsupported] should be False
        assert is_allowed_type(list[tuple]) is False
        assert is_allowed_type(list[set]) is False
        # dict[str, unsupported] should be False
        assert is_allowed_type(dict[str, tuple]) is False
        # Optional[unsupported] should be False
        assert is_allowed_type(Optional[tuple]) is False


class TestGetJsonSchemaForType:
    """Test suite for get_json_schema_for_type function."""

    def test_none_type(self):
        """Test JSON schema for None type."""
        assert get_json_schema_for_type(None) == {"type": "null"}
        assert get_json_schema_for_type(type(None)) == {"type": "null"}

    def test_simple_types(self):
        """Test JSON schema for simple types."""
        assert get_json_schema_for_type(str) == {"type": "string"}
        assert get_json_schema_for_type(int) == {"type": "integer"}
        assert get_json_schema_for_type(float) == {"type": "number"}
        assert get_json_schema_for_type(bool) == {"type": "boolean"}

    def test_pydantic_model(self):
        """Test JSON schema for Pydantic models."""
        schema = get_json_schema_for_type(SimpleModel)
        assert schema is not None
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_untyped_collections(self):
        """Test JSON schema for untyped collections."""
        assert get_json_schema_for_type(dict) == {"type": "object"}
        assert get_json_schema_for_type(list) == {"type": "array"}

    def test_typed_list(self):
        """Test JSON schema for typed lists."""
        schema = get_json_schema_for_type(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

        schema = get_json_schema_for_type(list[int])
        assert schema == {"type": "array", "items": {"type": "integer"}}

    def test_typed_dict(self):
        """Test JSON schema for typed dicts."""
        schema = get_json_schema_for_type(dict[str, str])
        assert schema == {"type": "object", "additionalProperties": {"type": "string"}}

        schema = get_json_schema_for_type(dict[str, int])
        assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_optional_types(self):
        """Test JSON schema for Optional types."""
        schema = get_json_schema_for_type(Optional[str])
        assert schema == {"anyOf": [{"type": "string"}, {"type": "null"}]}

        schema = get_json_schema_for_type(Optional[int])
        assert schema == {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    def test_nested_collections(self):
        """Test JSON schema for nested collections."""
        # list[list[str]]
        schema = get_json_schema_for_type(list[list[str]])
        assert schema == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        }

        # dict[str, list[int]]
        schema = get_json_schema_for_type(dict[str, list[int]])
        assert schema == {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "integer"}},
        }

    def test_optional_nested_types(self):
        """Test JSON schema for Optional nested types."""
        schema = get_json_schema_for_type(Optional[list[str]])
        assert schema == {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        }

    def test_unsupported_types(self):
        """Test that unsupported types return None."""
        assert get_json_schema_for_type(tuple) is None
        assert get_json_schema_for_type(set) is None
        assert get_json_schema_for_type(tuple[str, int]) is None

        # Complex unions
        from typing import Union

        assert get_json_schema_for_type(Union[str, int, float]) is None

        # Custom non-Pydantic classes
        class CustomClass:
            pass

        assert get_json_schema_for_type(CustomClass) is None

    def test_list_of_pydantic_models(self):
        """Test JSON schema for list of Pydantic models."""
        schema = get_json_schema_for_type(list[SimpleModel])
        assert schema is not None
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "object"
        assert "properties" in schema["items"]

    def test_dict_with_pydantic_model_values(self):
        """Test JSON schema for dict with Pydantic model values."""
        schema = get_json_schema_for_type(dict[str, SimpleModel])
        assert schema is not None
        assert schema["type"] == "object"
        assert schema["additionalProperties"]["type"] == "object"
        assert "properties" in schema["additionalProperties"]


class TestExtractSchemaFromType:
    """Test suite for extract_schema_from_type function."""

    def test_none_type_returns_schema(self):
        """Test that None type returns null schema."""
        result = extract_schema_from_type(None)
        assert result == {"type": "null"}

        result = extract_schema_from_type(type(None))
        assert result == {"type": "null"}

    def test_pydantic_model_returns_class(self):
        """Test that Pydantic models return the class itself."""
        result = extract_schema_from_type(SimpleModel)
        assert result is SimpleModel

        result = extract_schema_from_type(NestedModel)
        assert result is NestedModel

    def test_simple_types_return_schema(self):
        """Test that simple types return JSON schema dict."""
        result = extract_schema_from_type(str)
        assert isinstance(result, dict)
        assert result == {"type": "string"}

        result = extract_schema_from_type(int)
        assert isinstance(result, dict)
        assert result == {"type": "integer"}

    def test_collection_types_return_schema(self):
        """Test that collection types return JSON schema dict."""
        result = extract_schema_from_type(list[str])
        assert isinstance(result, dict)
        assert result["type"] == "array"

        result = extract_schema_from_type(dict[str, int])
        assert isinstance(result, dict)
        assert result["type"] == "object"

    def test_optional_types_return_schema(self):
        """Test that Optional types return JSON schema dict."""
        result = extract_schema_from_type(Optional[str])
        assert isinstance(result, dict)
        assert "anyOf" in result

    def test_unsupported_types_return_none(self):
        """Test that unsupported types return None."""
        assert extract_schema_from_type(tuple) is None
        assert extract_schema_from_type(set) is None
        assert extract_schema_from_type(tuple[str, int]) is None

        class CustomClass:
            pass

        assert extract_schema_from_type(CustomClass) is None

    def test_complex_allowed_types(self):
        """Test that complex allowed types return appropriate results."""
        # list[Pydantic] returns schema with nested model schema
        result = extract_schema_from_type(list[SimpleModel])
        assert isinstance(result, dict)
        assert result["type"] == "array"

        # Optional[Pydantic] returns schema
        result = extract_schema_from_type(Optional[SimpleModel])
        assert isinstance(result, dict)
        assert "anyOf" in result

    def test_nested_unsupported_types(self):
        """Test that nested unsupported types return None."""
        # list[tuple] is not allowed
        assert extract_schema_from_type(list[tuple]) is None

        # dict[str, set] is not allowed
        assert extract_schema_from_type(dict[str, set]) is None

        # Optional[tuple] is not allowed
        assert extract_schema_from_type(Optional[tuple]) is None


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_deeply_nested_structures(self):
        """Test deeply nested but valid structures."""
        # dict[str, list[dict[str, list[str]]]]
        deep_type = dict[str, list[dict[str, list[str]]]]
        assert is_allowed_type(deep_type) is True
        schema = get_json_schema_for_type(deep_type)
        assert schema is not None
        assert schema["type"] == "object"

    def test_optional_at_different_levels(self):
        """Test Optional at different nesting levels."""
        # Optional at top level
        assert is_allowed_type(Optional[str]) is True

        # Optional nested in list
        assert is_allowed_type(list[Optional[str]]) is True

        # Optional nested in dict
        assert is_allowed_type(dict[str, Optional[int]]) is True

    def test_multiple_pydantic_models(self):
        """Test with multiple different Pydantic models."""

        class Model1(BaseModel):
            value: str

        class Model2(BaseModel):
            count: int

        assert is_allowed_type(Model1) is True
        assert is_allowed_type(Model2) is True
        assert is_allowed_type(list[Model1]) is True
        assert is_allowed_type(dict[str, Model2]) is True

    def test_empty_types(self):
        """Test behavior with empty/unparameterized generics."""
        # These should be allowed as untyped collections
        assert is_allowed_type(list) is True
        assert is_allowed_type(dict) is True
        assert get_json_schema_for_type(list) == {"type": "array"}
        assert get_json_schema_for_type(dict) == {"type": "object"}

    def test_consistency_between_functions(self):
        """Test that all three functions work consistently together."""
        test_types = [
            str,
            int,
            list[str],
            dict[str, int],
            Optional[str],
            SimpleModel,
            tuple,  # unsupported
        ]

        for type_hint in test_types:
            is_allowed = is_allowed_type(type_hint)
            schema = get_json_schema_for_type(type_hint)
            extracted = extract_schema_from_type(type_hint)

            # If type is not allowed, schema functions should return None
            if not is_allowed:
                assert schema is None
                assert extracted is None
            else:
                # If type is allowed, schema functions should return something
                assert schema is not None or extracted is not None

    def test_none_type_variations(self):
        """Test different ways to express None type."""
        from typing import Union

        # Direct None
        assert is_allowed_type(None) is True

        # type(None)
        assert is_allowed_type(type(None)) is True

        # Optional[str] is Union[str, None]
        assert is_allowed_type(Optional[str]) is True

        # Explicit Union[T, None]
        assert is_allowed_type(Union[str, None]) is True

    def test_dict_key_validation(self):
        """Test that only str keys are allowed for dict."""
        # Valid
        assert is_allowed_type(dict[str, str]) is True

        # Invalid keys
        assert is_allowed_type(dict[int, str]) is False
        assert is_allowed_type(dict[float, str]) is False
        assert is_allowed_type(dict[bool, str]) is False
        assert is_allowed_type(dict[tuple, str]) is False
        assert is_allowed_type(dict[SimpleModel, str]) is False
