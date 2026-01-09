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

        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_register_simple_function.<locals>.simple_app"
        assert app_id in list_applications()
        app = get_application(app_id)
        assert app is not None
        assert app.input_type is None
        assert app.output_type is None

    def test_function_name_used_as_registry_key(self):
        """Test that function name is used as registry key."""

        @pixie_app
        def my_app(_input_data: JsonValue) -> JsonValue:
            return {"result": "ok"}

        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_function_name_used_as_registry_key.<locals>.my_app"
        assert app_id in list_applications()
        app = get_application(app_id)
        assert app is not None

    def test_register_duplicate_function_name_within_same_scope(self):
        """Test that registering duplicate function names in same scope overwrites."""

        @pixie_app
        def duplicate_app(_input_data: JsonValue) -> JsonValue:
            return {"app": "1"}

        # Second registration with same name in same scope overwrites
        @pixie_app
        def duplicate_app(_input_data: JsonValue) -> JsonValue:  # noqa # pylint: disable=E0102
            return {"app": "2"}

        # The second one should be registered
        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_register_duplicate_function_name_within_same_scope.<locals>.duplicate_app"
        apps = list_applications()
        # Both will have the same ID, so only one entry
        assert app_id in apps

    def test_decorator_with_parentheses(self):
        """Test using decorator with parentheses."""

        @pixie_app
        def app_with_parens(_input_data: JsonValue) -> JsonValue:
            return {"decorated": True}

        app_id = "tests.pixie.test_registry.TestBasicRegistration.test_decorator_with_parentheses.<locals>.app_with_parens"
        assert app_id in list_applications()


class TestTypeExtraction:
    """Test Pydantic type extraction."""

    def test_register_with_pydantic_input(self):
        """Test registering with Pydantic input type."""

        @pixie_app
        def typed_input_app(data: InputModel) -> JsonValue:
            return {"received": data.message}

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_pydantic_input.<locals>.typed_input_app"
        app = get_application(app_id)
        assert app is not None
        assert app.input_type == InputModel

    def test_register_with_pydantic_output(self):
        """Test registering with Pydantic output type."""

        @pixie_app
        def typed_output_app(_data: str) -> OutputModel:  # noqa: ARG001
            return OutputModel(result="success")

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_pydantic_output.<locals>.typed_output_app"
        app = get_application(app_id)
        assert app is not None
        assert app.output_type == OutputModel

    def test_register_with_both_pydantic_types(self):
        """Test registering with both Pydantic input and output."""

        @pixie_app
        def fully_typed_app(data: InputModel) -> OutputModel:
            return OutputModel(result=f"Processed: {data.message}")

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_both_pydantic_types.<locals>.fully_typed_app"
        app = get_application(app_id)
        assert app is not None
        assert app.input_type == InputModel
        assert app.output_type == OutputModel

    def test_register_with_simple_types(self):
        """Test registering with simple types like str, int."""

        @pixie_app
        def string_app(text: str) -> str:
            return text.upper()

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_simple_types.<locals>.string_app"
        app = get_application(app_id)
        assert app is not None
        assert isinstance(app.input_type, dict)
        assert app.input_type["type"] == "string"
        assert isinstance(app.output_type, dict)
        assert app.output_type["type"] == "string"

    def test_register_with_jsonvalue_returns_none_schema(self):
        """Test that JsonValue types result in None schema."""

        @pixie_app
        def json_app(data: JsonValue) -> JsonValue:
            return data

        app_id = "tests.pixie.test_registry.TestTypeExtraction.test_register_with_jsonvalue_returns_none_schema.<locals>.json_app"
        app = get_application(app_id)
        assert app is not None
        assert app.input_type is None
        assert app.output_type is None


class TestCallableApplications:
    """Test callable (non-generator) applications."""

    @pytest.mark.asyncio
    async def test_call_simple_sync_function(self):
        """Test calling a synchronous function."""

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
        async def my_generator(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"first": 1}
            yield {"second": 2}

        app_id = "tests.pixie.test_registry.TestGeneratorApplications.test_generator_type_detection.<locals>.my_generator"
        app = get_application(app_id)
        assert app is not None
        # Generator should be registered successfully
        assert app.stream_handler is not None


class TestGeneratorRegistration:
    """Test generator registration and schema extraction."""

    def test_generator_with_pydantic_yield_type_schema(self):
        """Test that generator yield type schema is extracted correctly."""

        @pixie_app
        async def pydantic_yield_gen(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[OutputModel, None]:
            yield OutputModel(result="test")

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_pydantic_yield_type_schema.<locals>.pydantic_yield_gen"
        app = get_application(app_id)
        assert app is not None
        assert app.output_type == OutputModel
        assert app.user_input_type is None  # No send type

    def test_generator_with_pydantic_send_type_schema(self):
        """Test that generator send type schema is extracted correctly."""

        @pixie_app
        async def interactive_gen(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[OutputModel, InputModel]:
            user_input: InputModel = yield OutputModel(result="first")  # type: ignore
            yield OutputModel(result=f"got: {user_input.message}")

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_pydantic_send_type_schema.<locals>.interactive_gen"
        app = get_application(app_id)
        assert app is not None
        assert app.output_type == OutputModel
        assert app.user_input_type == InputModel

    def test_generator_with_jsonvalue_yield_no_schema(self):
        """Test that JsonValue yield type results in None schema."""

        @pixie_app
        async def json_gen(
            _data: JsonValue,  # noqa: ARG001
        ) -> AsyncGenerator[JsonValue, None]:
            yield {"test": "data"}

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_jsonvalue_yield_no_schema.<locals>.json_gen"
        app = get_application(app_id)
        assert app is not None
        assert app.output_type is None  # JsonValue -> None
        assert app.user_input_type is None

    def test_generator_with_both_pydantic_types(self):
        """Test generator with both Pydantic input and output schemas."""

        @pixie_app
        async def full_typed_gen(
            data: InputModel,
        ) -> AsyncGenerator[OutputModel, InputModel]:
            yield OutputModel(result=f"Processing: {data.message}")
            feedback: InputModel = yield  # type: ignore
            yield OutputModel(result=f"Feedback: {feedback.message}")

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_both_pydantic_types.<locals>.full_typed_gen"
        app = get_application(app_id)
        assert app is not None
        assert app.input_type == InputModel
        assert app.output_type == OutputModel
        assert app.user_input_type == InputModel

    def test_generator_with_simple_type_input(self):
        """Test generator with simple type input has schema."""

        @pixie_app
        async def str_input_gen(text: str) -> AsyncGenerator[JsonValue, None]:
            yield {"processed": text.upper()}

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_with_simple_type_input.<locals>.str_input_gen"
        app = get_application(app_id)
        assert app is not None
        assert isinstance(app.input_type, dict)
        assert app.input_type["type"] == "string"

    def test_generator_factory_pattern_schema(self):
        """Test that async function returning generator extracts schema."""

        @pixie_app
        async def factory_with_types(
            data: InputModel,
        ) -> AsyncGenerator[OutputModel, None]:
            async def _gen() -> AsyncGenerator[OutputModel, None]:
                for i in range(data.count):
                    yield OutputModel(result=f"item_{i}")

            return _gen()

        app_id = "tests.pixie.test_registry.TestGeneratorRegistration.test_generator_factory_pattern_schema.<locals>.factory_with_types"
        app = get_application(app_id)
        assert app is not None
        assert app.input_type == InputModel
        assert app.output_type == OutputModel
        assert app.user_input_type is None

    @pytest.mark.asyncio
    async def test_interactive_generator_with_send(self):
        """Test interactive generator that receives user input."""

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        @pixie_app
        def app1(_data: JsonValue) -> JsonValue:  # noqa: ARG001
            return {}

        @pixie_app
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

        @pixie_app
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

        @pixie_app
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

        app_id = "tests.pixie.test_registry.TestComplexScenarios.test_streaming_large_dataset.<locals>.stream_numbers"
        result_stream = call_application(app_id, {"limit": 50})
        results = [item async for item in result_stream]

        assert len(results) == 50
        first_result = results[0]
        last_result = results[49]
        assert isinstance(first_result, dict) and first_result["number"] == 0
        assert isinstance(last_result, dict) and last_result["number"] == 49
