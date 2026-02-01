"""Test application registry functionality."""

# type: ignore - Tests intentionally use narrower types than the generic API accepts
# flake8: noqa: E501

import asyncio
import pytest
from typing import AsyncGenerator
from pydantic import BaseModel, JsonValue
from unittest.mock import Mock, patch

from pixie.registry import (
    app,
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

        @app
        def simple_app(input_data: JsonValue) -> JsonValue:
            return {"echo": input_data}

        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_register_simple_function.<locals>.simple_app"
        assert app_id in list_applications()
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type is None
        assert registered_app.output_type is None

    def test_function_name_used_as_registry_key(self):
        """Test that function name is used as registry key."""

        @app
        def my_app(_input_data: JsonValue) -> JsonValue:
            return {"result": "ok"}

        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_function_name_used_as_registry_key.<locals>.my_app"
        assert app_id in list_applications()
        registered_app = get_application(app_id)
        assert registered_app is not None

    def test_decorator_with_parentheses(self):
        """Test using decorator with parentheses."""

        @app
        def app_with_parens(_input_data: JsonValue) -> JsonValue:
            return {"decorated": True}

        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_decorator_with_parentheses.<locals>.app_with_parens"
        assert app_id in list_applications()


class TestTypeExtraction:
    """Test Pydantic type extraction."""

    def test_register_with_pydantic_input(self):
        """Test registering with Pydantic input type."""

        @app
        def typed_input_app(data: InputModel) -> JsonValue:
            return {"received": data.message}

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_pydantic_input.<locals>.typed_input_app"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type == InputModel

    def test_register_with_pydantic_output(self):
        """Test registering with Pydantic output type."""

        @app
        def typed_output_app(_data: str) -> OutputModel:  # noqa: ARG001
            return OutputModel(result="success")

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_pydantic_output.<locals>.typed_output_app"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.output_type == OutputModel

    def test_register_with_both_pydantic_types(self):
        """Test registering with both Pydantic input and output."""

        @app
        def fully_typed_app(data: InputModel) -> OutputModel:
            return OutputModel(result=f"Processed: {data.message}")

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_both_pydantic_types.<locals>.fully_typed_app"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type == InputModel
        assert registered_app.output_type == OutputModel

    def test_register_with_simple_types(self):
        """Test registering with simple types like str, int."""

        @app
        def string_app(text: str) -> str:
            return text.upper()

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_simple_types.<locals>.string_app"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert isinstance(registered_app.input_type, dict)
        assert registered_app.input_type["type"] == "string"
        assert isinstance(registered_app.output_type, dict)
        assert registered_app.output_type["type"] == "string"

    def test_register_with_jsonvalue_returns_none_schema(self):
        """Test that JsonValue types result in None schema."""

        @app
        def json_app(data: JsonValue) -> JsonValue:
            return data

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_jsonvalue_returns_none_schema.<locals>.json_app"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type is None
        assert registered_app.output_type is None


class TestCallableApplications:
    """Test callable (non-generator) applications."""

    @pytest.mark.asyncio
    async def test_call_simple_sync_function(self):
        """Test calling a synchronous function."""

        @app
        def sync_app(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {"result": "sync"}

        app_id = "tests.pixie.test_registry.TestCallableApplications.test_call_simple_sync_function.<locals>.sync_app"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"result": "sync"}

    @pytest.mark.asyncio
    async def test_call_async_function(self):
        """Test calling an async function."""

        @app
        async def async_app(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            await asyncio.sleep(0.001)
            return {"result": "async"}

        app_id = "tests.pixie.test_registry.TestCallableApplications.test_call_async_function.<locals>.async_app"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"result": "async"}

    @pytest.mark.asyncio
    async def test_call_with_pydantic_input(self):
        """Test calling with Pydantic input model."""

        @app
        def pydantic_input_app(data: InputModel) -> JsonValue:
            return {"message": data.message, "count": data.count}

        app_id = "tests.pixie.test_registry.TestCallableApplications.test_call_with_pydantic_input.<locals>.pydantic_input_app"
        result_stream = call_application(app_id, {"message": "hello", "count": 5})
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"message": "hello", "count": 5}

    @pytest.mark.asyncio
    async def test_call_with_pydantic_output(self):
        """Test that Pydantic output is automatically serialized."""

        @app
        def pydantic_output_app(_data: JsonValue) -> OutputModel:  # noqa: ARG001
            return OutputModel(result="success", processed=True)

        app_id = "tests.pixie.test_registry.TestCallableApplications.test_call_with_pydantic_output.<locals>.pydantic_output_app"
        result_stream = call_application(app_id, {})
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

        @app
        async def generator_app(
            _data: JsonValue,
        ) -> AsyncGenerator[JsonValue, None]:  # noqa: ARG001
            for i in range(3):
                yield {"item": i}

        app_id = "tests.pixie.test_registry.TestGeneratorApplications.test_async_generator_function.<locals>.generator_app"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 3
        assert results[0] == {"item": 0}
        assert results[1] == {"item": 1}
        assert results[2] == {"item": 2}

    @pytest.mark.asyncio
    async def test_async_generator_factory(self):
        """Test async function returning generator (Pattern 2)."""

        @app
        async def factory_app(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            async def _generator() -> AsyncGenerator[JsonValue, None]:
                for i in range(2):
                    yield {"value": i * 10}

            return _generator()

        app_id = "tests.pixie.test_registry.TestGeneratorApplications.test_async_generator_factory.<locals>.factory_app"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 2
        assert results[0] == {"value": 0}
        assert results[1] == {"value": 10}

    @pytest.mark.asyncio
    async def test_generator_with_pydantic_input(self):
        """Test generator with Pydantic input model."""

        @app
        async def typed_generator(
            data: InputModel,
        ) -> AsyncGenerator[JsonValue, None]:
            for i in range(data.count):
                yield {"message": data.message, "iteration": i}

        app_id = "tests.pixie.test_registry.TestGeneratorApplications.test_generator_with_pydantic_input.<locals>.typed_generator"
        result_stream = call_application(app_id, {"message": "test", "count": 3})
        results = [item async for item in result_stream]

        assert len(results) == 3
        assert all(item["message"] == "test" for item in results)  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_generator_with_pydantic_output(self):
        """Test generator yielding Pydantic models."""

        @app
        async def pydantic_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[OutputModel, None]:
            for i in range(2):
                yield OutputModel(result=f"item_{i}", processed=True)

        app_id = "tests.pixie.test_registry.TestGeneratorApplications.test_generator_with_pydantic_output.<locals>.pydantic_generator"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 2
        assert results[0] == {"result": "item_0", "processed": True}
        assert results[1] == {"result": "item_1", "processed": True}

    @pytest.mark.asyncio
    async def test_generator_type_detection(self):
        """Test that generator is properly detected."""

        @app
        async def my_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"first": 1}
            yield {"second": 2}

        app_id = "tests.pixie.test_registry.TestGeneratorApplications.test_generator_type_detection.<locals>.my_generator"
        pixie_app = get_application(app_id)
        assert pixie_app is not None
        # Generator should be registered successfully
        assert pixie_app.stream_handler is not None


class TestGeneratorRegistration:
    """Test generator registration and schema extraction."""

    def test_generator_with_pydantic_yield_type_schema(self):
        """Test that generator yield type schema is extracted correctly."""

        @app
        async def pydantic_yield_gen(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[OutputModel, None]:
            yield OutputModel(result="test")

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_pydantic_yield_type_schema.<locals>.pydantic_yield_gen"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.output_type == OutputModel
        assert registered_app.user_input_type is None  # No send type

    def test_generator_with_pydantic_send_type_schema(self):
        """Test that generator send type schema is extracted correctly."""

        @app
        async def interactive_gen(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[OutputModel, InputModel]:
            user_input: InputModel = yield OutputModel(result="first")  # type: ignore
            yield OutputModel(result=f"got: {user_input.message}")

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_pydantic_send_type_schema.<locals>.interactive_gen"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.output_type == OutputModel
        assert registered_app.user_input_type == InputModel

    def test_generator_with_jsonvalue_yield_no_schema(self):
        """Test that JsonValue yield type results in None schema."""

        @app
        async def json_gen(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"test": "data"}

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_jsonvalue_yield_no_schema.<locals>.json_gen"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.output_type is None  # JsonValue -> None
        assert registered_app.user_input_type is None

    def test_generator_with_both_pydantic_types(self):
        """Test generator with both Pydantic input and output schemas."""

        @app
        async def full_typed_gen(
            data: InputModel,
        ) -> AsyncGenerator[OutputModel, InputModel]:
            yield OutputModel(result=f"Processing: {data.message}")
            feedback: InputModel = yield  # type: ignore
            yield OutputModel(result=f"Feedback: {feedback.message}")

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_both_pydantic_types.<locals>.full_typed_gen"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type == InputModel
        assert registered_app.output_type == OutputModel
        assert registered_app.user_input_type == InputModel

    def test_generator_with_simple_type_input(self):
        """Test generator with simple type input has schema."""

        @app
        async def str_input_gen(text: str) -> AsyncGenerator[JsonValue, None]:
            yield {"processed": text.upper()}

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_simple_type_input.<locals>.str_input_gen"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert isinstance(registered_app.input_type, dict)
        assert registered_app.input_type["type"] == "string"

    def test_generator_factory_pattern_schema(self):
        """Test that async function returning generator extracts schema."""

        @app
        async def factory_with_types(
            data: InputModel,
        ) -> AsyncGenerator[OutputModel, None]:
            async def _gen() -> AsyncGenerator[OutputModel, None]:
                for i in range(data.count):
                    yield OutputModel(result=f"item_{i}")

            return _gen()

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_factory_pattern_schema.<locals>.factory_with_types"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type == InputModel
        assert registered_app.output_type == OutputModel
        assert registered_app.user_input_type is None

    @pytest.mark.asyncio
    async def test_interactive_generator_with_send(self):
        """Test interactive generator that receives user input."""

        @app
        async def interactive(
            data: InputModel,
        ) -> AsyncGenerator[OutputModel, InputModel]:
            # First yield - output to user
            yield OutputModel(result=f"Starting: {data.message}")

            # Second yield - waiting for user input (yields None)
            user_input: InputModel = yield  # type: ignore

            # Third yield - output based on user input
            yield OutputModel(result=f"User said: {user_input.message}")

        # Call and interact with the generator
        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_interactive_generator_with_send.<locals>.interactive"
        gen = call_application(app_id, {"message": "hello", "count": 1})

        # Get first output
        first = await gen.asend(None)
        assert first == {"result": "Starting: hello"}

        # Generator is now at the "waiting for input" yield, which returns None
        waiting = await gen.asend(None)
        assert waiting is None

        # Send user input and get final output
        final = await gen.asend({"message": "feedback", "count": 2})
        assert final == {"result": "User said: feedback"}

        # Generator should complete
        try:
            await gen.asend(None)
            assert False, "Expected StopAsyncIteration"
        except StopAsyncIteration:
            pass


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_generator(self):
        """Test generator that yields nothing."""

        @app
        async def empty_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            return
            yield  # pragma: no cover

        app_id = "tests.pixie.test_registry.TestEdgeCases.test_empty_generator.<locals>.empty_generator"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_generator_with_await_inside(self):
        """Test generator with async operations inside."""

        @app
        async def async_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            for i in range(2):
                await asyncio.sleep(0.001)
                yield {"delayed": i}

        app_id = "tests.pixie.test_registry.TestEdgeCases.test_generator_with_await_inside.<locals>.async_generator"
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_mixed_pydantic_and_json_values(self):
        """Test returning mixed Pydantic models and plain JSON."""

        @app
        def mixed_app(data: JsonValue) -> JsonValue | OutputModel:
            # Sometimes return Pydantic, sometimes plain JSON
            if isinstance(data, dict) and data.get("use_pydantic"):
                return OutputModel(result="pydantic")
            return {"result": "json"}

        app_id = "tests.pixie.test_registry.TestEdgeCases.test_mixed_pydantic_and_json_values.<locals>.mixed_app"
        # Test with plain JSON
        results = [item async for item in call_application(app_id, {})]
        assert results[0] == {"result": "json"}

        # Test with Pydantic - model_dump excludes defaults by default with exclude_unset=True
        results = [
            item async for item in call_application(app_id, {"use_pydantic": True})
        ]
        result_dict = results[0]
        assert isinstance(result_dict, dict)
        assert result_dict["result"] == "pydantic"

    def test_list_applications(self):
        """Test listing all registered applications."""

        @app
        def app1(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        @app
        def app2(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        apps = list_applications()
        assert len(apps) == 2
        app1_id = "tests.pixie.test_registry.TestEdgeCases.test_list_applications.<locals>.app1"
        app2_id = "tests.pixie.test_registry.TestEdgeCases.test_list_applications.<locals>.app2"
        assert app1_id in apps
        assert app2_id in apps

    def test_clear_registry(self):
        """Test clearing the registry."""

        @app
        def temp_app(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        assert len(list_applications()) == 1

        clear_registry()

        assert len(list_applications()) == 0
        temp_app_id = "tests.pixie.test_registry.TestEdgeCases.test_clear_registry.<locals>.temp_app"
        assert get_application(temp_app_id) is None


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    @pytest.mark.asyncio
    async def test_chained_processing(self):
        """Test processing data through multiple steps."""

        @app
        async def processor(data: InputModel) -> OutputModel:
            # Simulate processing
            await asyncio.sleep(0.001)
            return OutputModel(result=f"Processed {data.count} times: {data.message}")

        app_id = "tests.pixie.test_registry.TestComplexScenarios.test_chained_processing.<locals>.processor"
        result_stream = call_application(app_id, {"message": "hello", "count": 3})
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

        @app
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

        app_id = "tests.pixie.test_registry.TestComplexScenarios.test_streaming_large_dataset.<locals>.stream_numbers"
        result_stream = call_application(app_id, {"limit": 50})
        results = [item async for item in result_stream]

        assert len(results) == 50
        first_result = results[0]
        last_result = results[49]
        assert isinstance(first_result, dict) and first_result["number"] == 0
        assert isinstance(last_result, dict) and last_result["number"] == 49


class TestNoArgHandlers:
    """Test handlers with no arguments."""

    def test_register_sync_no_arg_callable(self):
        """Test registering a sync function with no arguments."""

        @app
        def no_arg_sync() -> JsonValue:
            return {"result": "no args needed"}

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_register_sync_no_arg_callable.<locals>.no_arg_sync"
        assert app_id in list_applications()
        registered_app = get_application(app_id)
        assert registered_app is not None
        # Should have null schema since no args means None type
        assert registered_app.input_type == {"type": "null"}
        # Output type may be None for JsonValue return type (no specific schema)

    def test_register_async_no_arg_callable(self):
        """Test registering an async function with no arguments."""

        @app
        async def no_arg_async() -> JsonValue:
            return {"result": "async no args"}

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_register_async_no_arg_callable.<locals>.no_arg_async"
        assert app_id in list_applications()
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type == {"type": "null"}

    def test_register_no_arg_with_pydantic_output(self):
        """Test registering no-arg function with Pydantic output."""

        @app
        async def no_arg_pydantic() -> OutputModel:
            return OutputModel(result="success", processed=True)

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_register_no_arg_with_pydantic_output.<locals>.no_arg_pydantic"
        registered_app = get_application(app_id)
        assert registered_app is not None
        assert registered_app.input_type == {"type": "null"}
        assert registered_app.output_type == OutputModel

    @pytest.mark.asyncio
    async def test_call_sync_no_arg_callable(self):
        """Test calling a sync function with no arguments."""

        @app
        def no_arg_func() -> JsonValue:
            return {"message": "Hello, World!"}

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_call_sync_no_arg_callable.<locals>.no_arg_func"
        # Input data is ignored for no-arg functions
        result_stream = call_application(app_id, None)
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"message": "Hello, World!"}

    @pytest.mark.asyncio
    async def test_call_async_no_arg_callable(self):
        """Test calling an async function with no arguments."""

        @app
        async def no_arg_async() -> JsonValue:
            await asyncio.sleep(0.01)  # Simulate async work
            return {"status": "done"}

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_call_async_no_arg_callable.<locals>.no_arg_async"
        result_stream = call_application(app_id, None)
        results = [item async for item in result_stream]

        assert len(results) == 1
        assert results[0] == {"status": "done"}

    @pytest.mark.asyncio
    async def test_call_no_arg_with_pydantic_output(self):
        """Test calling no-arg function that returns Pydantic model."""

        @app
        async def no_arg_model() -> OutputModel:
            return OutputModel(result="generated", processed=True)

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_call_no_arg_with_pydantic_output.<locals>.no_arg_model"
        result_stream = call_application(app_id, None)
        results = [item async for item in result_stream]

        assert len(results) == 1
        # Pydantic model should be serialized
        assert results[0] == {"result": "generated", "processed": True}

    @pytest.mark.asyncio
    async def test_no_arg_generator(self):
        """Test no-arg async generator function."""

        @app
        async def no_arg_gen() -> AsyncGenerator[JsonValue, None]:
            for i in range(3):
                yield {"count": i}

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_no_arg_generator.<locals>.no_arg_gen"
        result_stream = call_application(app_id, None)
        results = [item async for item in result_stream]

        assert len(results) == 3
        assert results[0] == {"count": 0}
        assert results[1] == {"count": 1}
        assert results[2] == {"count": 2}

    @pytest.mark.asyncio
    async def test_no_arg_generator_with_pydantic_yield(self):
        """Test no-arg generator yielding Pydantic models."""

        @app
        async def no_arg_pydantic_gen() -> AsyncGenerator[OutputModel, None]:
            for i in range(2):
                yield OutputModel(result=f"item_{i}", processed=True)

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_no_arg_generator_with_pydantic_yield.<locals>.no_arg_pydantic_gen"
        result_stream = call_application(app_id, None)
        results = [item async for item in result_stream]

        assert len(results) == 2
        assert results[0] == {"result": "item_0", "processed": True}
        assert results[1] == {"result": "item_1", "processed": True}

    @pytest.mark.asyncio
    async def test_no_arg_callable_input_ignored(self):
        """Test that input data is ignored for no-arg functions."""

        @app
        def no_arg_ignores_input() -> JsonValue:
            return {"constant": "value"}

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_no_arg_callable_input_ignored.<locals>.no_arg_ignores_input"

        # Try with different input values - all should work and return same result
        for test_input in [None, {}, {"ignored": "data"}, "string", 123]:
            result_stream = call_application(app_id, test_input)
            results = [item async for item in result_stream]
            assert results[0] == {"constant": "value"}

    @pytest.mark.asyncio
    async def test_no_arg_generator_factory(self):
        """Test no-arg async function returning generator."""

        @app
        async def no_arg_factory() -> AsyncGenerator[JsonValue, None]:
            async def _generator() -> AsyncGenerator[JsonValue, None]:
                yield {"step": 1}
                yield {"step": 2}

            return _generator()

        app_id = "tests.pixie.test_registry.TestNoArgHandlers.test_no_arg_generator_factory.<locals>.no_arg_factory"
        result_stream = call_application(app_id, None)
        results = [item async for item in result_stream]

        assert len(results) == 2
        assert results[0] == {"step": 1}
        assert results[1] == {"step": 2}


class TestObservationLogic:
    """Test the newly added observation enter/exit logic."""

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_callable_observation_started_and_ended(self, mock_langfuse):
        """Test that observations are properly started and ended for callable applications."""
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=None)
        mock_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        @app
        def simple_callable(_input_data: JsonValue) -> JsonValue:
            return {"result": "success"}

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_callable_observation_started_and_ended.<locals>.simple_callable"

        # Call the application
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        # Verify results
        assert len(results) == 1
        assert results[0] == {"result": "success"}

        # Verify observation was started with correct parameters
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            name="simple_callable", as_type="chain"
        )

        # Verify span was entered and exited properly
        mock_span.__enter__.assert_called_once()
        mock_span.__exit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_callable_observation_exception_handling(self, mock_langfuse):
        """Test that exceptions in callable applications properly exit observations."""
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=None)
        mock_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        @app
        def failing_callable(_input_data: JsonValue) -> JsonValue:
            raise ValueError("Test error")

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_callable_observation_exception_handling.<locals>.failing_callable"

        # Call the application and expect exception
        result_stream = call_application(app_id, {})
        with pytest.raises(ValueError, match="Test error"):
            await result_stream.__anext__()

        # Verify observation was started
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            name="failing_callable", as_type="chain"
        )

        # Verify span was entered
        mock_span.__enter__.assert_called_once()

        # Verify span was exited with exception info
        mock_span.__exit__.assert_called_once()
        call_args = mock_span.__exit__.call_args
        assert call_args[0][0] == ValueError  # exc_type
        assert isinstance(call_args[0][1], ValueError)  # exc_value
        assert call_args[0][2] is not None  # traceback

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_generator_observation_started_and_ended(self, mock_langfuse):
        """Test that observations are properly started and ended for generator applications."""
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=None)
        mock_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        @app
        async def simple_generator(
            _input_data: JsonValue,
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"step": 1}
            yield {"step": 2}

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_generator_observation_started_and_ended.<locals>.simple_generator"

        # Call the application
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        # Verify results
        assert len(results) == 2
        assert results[0] == {"step": 1}
        assert results[1] == {"step": 2}

        # Verify observation was started with correct parameters
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            name="simple_generator", as_type="chain"
        )

        # Verify span was entered and exited properly
        mock_span.__enter__.assert_called_once()
        mock_span.__exit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_generator_observation_exception_handling(self, mock_langfuse):
        """Test that exceptions in generator applications properly exit observations."""
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=None)
        mock_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        @app
        async def failing_generator(
            _input_data: JsonValue,
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"step": 1}
            raise RuntimeError("Generator error")

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_generator_observation_exception_handling.<locals>.failing_generator"

        # Call the application and expect exception
        result_stream = call_application(app_id, {})
        # Get first result
        result1 = await result_stream.__anext__()
        assert result1 == {"step": 1}
        # Second iteration should raise exception
        with pytest.raises(RuntimeError, match="Generator error"):
            await result_stream.__anext__()

        # Verify observation was started
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            name="failing_generator", as_type="chain"
        )

        # Verify span was entered
        mock_span.__enter__.assert_called_once()

        # Verify span was exited with exception info
        mock_span.__exit__.assert_called_once()
        call_args = mock_span.__exit__.call_args
        assert call_args[0][0] == RuntimeError  # exc_type
        assert isinstance(call_args[0][1], RuntimeError)  # exc_value
        assert call_args[0][2] is not None  # traceback

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_input_required_creates_wait_observation(self, mock_langfuse):
        """Test that yielding InputRequired creates a 'wait_of_input' observation."""
        from pixie.types import InputRequired

        mock_chain_span = Mock()
        mock_chain_span.__enter__ = Mock(return_value=None)
        mock_chain_span.__exit__ = Mock(return_value=None)
        mock_tool_span = Mock()
        mock_tool_span.__enter__ = Mock(return_value=None)
        mock_tool_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.side_effect = [
            mock_chain_span,
            mock_tool_span,
        ]

        @app
        async def interactive_generator(
            _input_data: JsonValue,
        ) -> AsyncGenerator[JsonValue | InputRequired, str]:
            yield {"message": "Enter your name"}
            user_input = yield InputRequired(str)
            yield {"greeting": f"Hello, {user_input}!"}

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_input_required_creates_wait_observation.<locals>.interactive_generator"

        # Call the application
        result_stream = call_application(app_id, {})

        # Get first result (should be the message)
        result1 = await result_stream.__anext__()
        assert result1 == {"message": "Enter your name"}

        # Get second result (should be InputRequired)
        result2 = await result_stream.__anext__()
        assert isinstance(result2, InputRequired)

        # Send user input and get greeting
        result3 = await result_stream.asend("Alice")
        assert result3 == {"greeting": "Hello, Alice!"}

        # Exhaust the generator
        with pytest.raises(StopAsyncIteration):
            await result_stream.__anext__()

        # Verify chain observation was started
        assert mock_langfuse.start_as_current_observation.call_count >= 1
        chain_call = mock_langfuse.start_as_current_observation.call_args_list[0]
        assert chain_call[1] == {"name": "interactive_generator", "as_type": "chain"}

        # Verify tool observations were started for InputRequired yields
        tool_calls = [
            call
            for call in mock_langfuse.start_as_current_observation.call_args_list[1:]
            if call[1]["as_type"] == "tool"
        ]
        assert len(tool_calls) >= 1  # At least one tool observation
        assert all(
            call[1] == {"name": "wait_of_input", "as_type": "tool"}
            for call in tool_calls
        )

        # Verify spans were managed properly
        mock_chain_span.__enter__.assert_called_once()
        mock_chain_span.__exit__.assert_called_once_with(None, None, None)

        # Verify tool spans were managed
        assert mock_tool_span.__enter__.call_count >= 1
        assert mock_tool_span.__exit__.call_count >= 1

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_no_input_required_does_not_create_wait_observation(
        self, mock_langfuse
    ):
        """Test that yielding non-InputRequired values does not create wait observations."""
        mock_chain_span = Mock()
        mock_chain_span.__enter__ = Mock(return_value=None)
        mock_chain_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_chain_span

        @app
        async def simple_generator(
            _input_data: JsonValue,
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"step": 1}
            yield {"step": 2}

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_no_input_required_does_not_create_wait_observation.<locals>.simple_generator"

        # Call the application
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        # Verify results
        assert len(results) == 2

        # Verify only chain observation was started (no tool observations)
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            name="simple_generator", as_type="chain"
        )

    @pytest.mark.asyncio
    @patch("pixie.registry._langfuse")
    async def test_async_callable_observation(self, mock_langfuse):
        """Test observation logic for async callable applications."""
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=None)
        mock_span.__exit__ = Mock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        @app
        async def async_callable(_input_data: JsonValue) -> JsonValue:
            await asyncio.sleep(0.001)  # Simulate async work
            return {"async_result": "done"}

        app_id = "tests.pixie.test_registry.TestObservationLogic.test_async_callable_observation.<locals>.async_callable"

        # Call the application
        result_stream = call_application(app_id, {})
        results = [item async for item in result_stream]

        # Verify results
        assert len(results) == 1
        assert results[0] == {"async_result": "done"}

        # Verify observation was managed
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            name="async_callable", as_type="chain"
        )
        mock_span.__enter__.assert_called_once()
        mock_span.__exit__.assert_called_once_with(None, None, None)
