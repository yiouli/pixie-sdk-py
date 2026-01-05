"""Test application registry functionality."""

# type: ignore - Tests intentionally use narrower types than the generic API accepts

import asyncio
import pytest
from typing import AsyncGenerator
from pydantic import BaseModel, JsonValue

from pixie.registry import (
    pixie_app,
    call_application,
    get_application,
    list_applications,
    clear_registry,
)


# Test models
class InputModel(BaseModel):
    message: str
    count: int = 1


class OutputModel(BaseModel):
    result: str
    processed: bool = True


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestBasicRegistration:
    """Test basic application registration."""

    def test_register_simple_function(self):
        """Test registering a simple function with no type hints."""

        @pixie_app
        def simple_app(input_data: JsonValue) -> JsonValue:
            return {"echo": input_data}

        assert "simple_app" in list_applications()
        app = get_application("simple_app")
        assert app is not None
        assert app.input_type is None
        assert app.output_type is None

    def test_register_with_custom_name(self):
        """Test registering with a custom name."""

        @pixie_app(name="custom_name")
        def my_app(_input_data: JsonValue) -> JsonValue:
            return {"result": "ok"}

        assert "custom_name" in list_applications()
        assert "my_app" not in list_applications()

    def test_register_duplicate_name_raises_error(self):
        """Test that registering duplicate names raises an error."""

        @pixie_app(name="duplicate")
        def app1(_input_data: JsonValue) -> JsonValue:
            return {"app": "1"}

        with pytest.raises(ValueError, match="already registered"):

            @pixie_app(name="duplicate")
            def app2(_input_data: JsonValue) -> JsonValue:
                return {"app": "2"}

    def test_decorator_with_parentheses(self):
        """Test using decorator with parentheses."""

        @pixie_app
        def app_with_parens(_input_data: JsonValue) -> JsonValue:
            return {"decorated": True}

        assert "app_with_parens" in list_applications()


class TestTypeExtraction:
    """Test Pydantic type extraction."""

    def test_register_with_pydantic_input(self):
        """Test registering with Pydantic input type."""

        @pixie_app
        def typed_input_app(data: InputModel) -> JsonValue:
            return {"received": data.message}

        app = get_application("typed_input_app")
        assert app is not None
        assert app.input_type == InputModel

    def test_register_with_pydantic_output(self):
        """Test registering with Pydantic output type."""

        @pixie_app
        def typed_output_app(_data: str) -> OutputModel:  # noqa: ARG001
            return OutputModel(result="success")

        app = get_application("typed_output_app")
        assert app is not None
        assert app.output_type == OutputModel

    def test_register_with_both_pydantic_types(self):
        """Test registering with both Pydantic input and output."""

        @pixie_app
        def fully_typed_app(data: InputModel) -> OutputModel:
            return OutputModel(result=f"Processed: {data.message}")

        app = get_application("fully_typed_app")
        assert app is not None
        assert app.input_type == InputModel
        assert app.output_type == OutputModel


class TestCallableApplications:
    """Test callable (non-generator) applications."""

    @pytest.mark.asyncio
    async def test_call_simple_sync_function(self):
        """Test calling a synchronous function."""

        @pixie_app
        def sync_app(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {"result": "sync"}

        result_stream = call_application("sync_app", {})
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"result": "sync"}

    @pytest.mark.asyncio
    async def test_call_async_function(self):
        """Test calling an async function."""

        @pixie_app
        async def async_app(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            await asyncio.sleep(0.001)
            return {"result": "async"}

        result_stream = call_application("async_app", {})
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"result": "async"}

    @pytest.mark.asyncio
    async def test_call_with_pydantic_input(self):
        """Test calling with Pydantic input model."""

        @pixie_app
        def pydantic_input_app(data: InputModel) -> JsonValue:
            return {"message": data.message, "count": data.count}

        result_stream = call_application(
            "pydantic_input_app", {"message": "hello", "count": 5}
        )
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"message": "hello", "count": 5}

    @pytest.mark.asyncio
    async def test_call_with_pydantic_output(self):
        """Test that Pydantic output is automatically serialized."""

        @pixie_app
        def pydantic_output_app(_data: JsonValue) -> OutputModel:  # noqa: ARG001
            return OutputModel(result="success", processed=True)

        result_stream = call_application("pydantic_output_app", {})
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"result": "success", "processed": True}

    @pytest.mark.asyncio
    async def test_call_nonexistent_app_raises_error(self):
        """Test that calling non-existent app raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            async for _item in call_application("nonexistent", {}):
                pass


class TestGeneratorApplications:
    """Test generator applications."""

    @pytest.mark.asyncio
    async def test_async_generator_function(self):
        """Test async generator function (Pattern 1)."""

        @pixie_app
        async def generator_app(
            _data: JsonValue,
        ) -> AsyncGenerator[JsonValue, None]:  # noqa: ARG001
            for i in range(3):
                yield {"item": i}

        result_stream = call_application("generator_app", {})
        results = [item async for item in result_stream]

        assert len(results) == 3
        assert results[0] == {"item": 0}
        assert results[1] == {"item": 1}
        assert results[2] == {"item": 2}

    @pytest.mark.asyncio
    async def test_async_generator_factory(self):
        """Test async function returning generator (Pattern 2)."""

        @pixie_app
        async def factory_app(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            async def _generator() -> AsyncGenerator[JsonValue, None]:
                for i in range(2):
                    yield {"value": i * 10}

            return _generator()

        result_stream = call_application("factory_app", {})
        results = [item async for item in result_stream]

        assert len(results) == 2
        assert results[0] == {"value": 0}
        assert results[1] == {"value": 10}

    @pytest.mark.asyncio
    async def test_generator_with_pydantic_input(self):
        """Test generator with Pydantic input model."""

        @pixie_app
        async def typed_generator(
            data: InputModel,
        ) -> AsyncGenerator[JsonValue, None]:
            for i in range(data.count):
                yield {"message": data.message, "iteration": i}

        result_stream = call_application(
            "typed_generator", {"message": "test", "count": 3}
        )
        results = [item async for item in result_stream]

        assert len(results) == 3
        assert all(item["message"] == "test" for item in results)  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_generator_with_pydantic_output(self):
        """Test generator yielding Pydantic models."""

        @pixie_app
        async def pydantic_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[OutputModel, None]:
            for i in range(2):
                yield OutputModel(result=f"item_{i}", processed=True)

        result_stream = call_application("pydantic_generator", {})
        results = [item async for item in result_stream]

        assert len(results) == 2
        assert results[0] == {"result": "item_0", "processed": True}
        assert results[1] == {"result": "item_1", "processed": True}

    @pytest.mark.asyncio
    async def test_generator_type_detection(self):
        """Test that generator is properly detected."""

        @pixie_app
        async def my_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"first": 1}
            yield {"second": 2}

        app = get_application("my_generator")
        assert app is not None
        # Generator should be registered successfully
        assert app.stream_handler is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_generator(self):
        """Test generator that yields nothing."""

        @pixie_app
        async def empty_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            return
            yield  # pragma: no cover

        result_stream = call_application("empty_generator", {})
        results = [item async for item in result_stream]

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_generator_with_await_inside(self):
        """Test generator with async operations inside."""

        @pixie_app
        async def async_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            for i in range(2):
                await asyncio.sleep(0.001)
                yield {"delayed": i}

        result_stream = call_application("async_generator", {})
        results = [item async for item in result_stream]

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_mixed_pydantic_and_json_values(self):
        """Test returning mixed Pydantic models and plain JSON."""

        @pixie_app
        def mixed_app(data: JsonValue) -> JsonValue | OutputModel:
            # Sometimes return Pydantic, sometimes plain JSON
            if isinstance(data, dict) and data.get("use_pydantic"):
                return OutputModel(result="pydantic")
            return {"result": "json"}

        # Test with plain JSON
        results = [item async for item in call_application("mixed_app", {})]
        assert results[0] == {"result": "json"}

        # Test with Pydantic - model_dump excludes defaults by default with exclude_unset=True
        results = [
            item async for item in call_application("mixed_app", {"use_pydantic": True})
        ]
        result_dict = results[0]
        assert isinstance(result_dict, dict)
        assert result_dict["result"] == "pydantic"

    def test_list_applications(self):
        """Test listing all registered applications."""

        @pixie_app
        def app1(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        @pixie_app
        def app2(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        apps = list_applications()
        assert len(apps) == 2
        assert "app1" in apps
        assert "app2" in apps

    def test_clear_registry(self):
        """Test clearing the registry."""

        @pixie_app
        def temp_app(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        assert len(list_applications()) == 1

        clear_registry()

        assert len(list_applications()) == 0
        assert get_application("temp_app") is None


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    @pytest.mark.asyncio
    async def test_chained_processing(self):
        """Test processing data through multiple steps."""

        @pixie_app
        async def processor(data: InputModel) -> OutputModel:
            # Simulate processing
            await asyncio.sleep(0.001)
            return OutputModel(result=f"Processed {data.count} times: {data.message}")

        result_stream = call_application("processor", {"message": "hello", "count": 3})
        results = [item async for item in result_stream]

        assert len(results) == 1
        result_dict = results[0]
        assert isinstance(result_dict, dict)
        result_value = result_dict["result"]
        assert (
            isinstance(result_value, str) and "Processed 3 times: hello" in result_value
        )

    @pytest.mark.asyncio
    async def test_streaming_large_dataset(self):
        """Test streaming a large number of items."""

        @pixie_app
        async def stream_numbers(
            data: JsonValue,
        ) -> AsyncGenerator[JsonValue, None]:
            limit_value = data.get("limit", 100) if isinstance(data, dict) else 100
            # Ensure limit is an int
            if isinstance(limit_value, int):
                limit = limit_value
            elif isinstance(limit_value, float):
                limit = int(limit_value)
            else:
                limit = 100
            for i in range(limit):
                yield {"number": i}

        result_stream = call_application("stream_numbers", {"limit": 50})
        results = [item async for item in result_stream]

        assert len(results) == 50
        first_result = results[0]
        last_result = results[49]
        assert isinstance(first_result, dict) and first_result["number"] == 0
        assert isinstance(last_result, dict) and last_result["number"] == 49
