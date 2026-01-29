# type: ignore
"""Test span processor prompt and trace emission functionality."""

import asyncio
import pytest
from unittest.mock import Mock, patch


from opentelemetry.trace import format_span_id, format_trace_id

import pixie.execution_context as exec_ctx
from pixie.prompts.prompt import BasePrompt, Variables
from pixie.types import PromptForSpan


class PromptTestVars(Variables):
    """Test variables class for prompt testing."""

    name: str
    count: int


@pytest.mark.asyncio
async def test_emit_prompt_attributes_with_compiled_prompt():
    """Test that _emit_prompt_attributes emits prompt info when span contains compiled prompt."""

    # Initialize execution context
    run_id = "test-run-123"
    exec_ctx.init_run(run_id)

    # Create a prompt and compile it
    prompt = BasePrompt(versions="Test prompt without variables")
    compiled_text = prompt.compile()

    # Mock span with the compiled text as an attribute
    mock_span = Mock()
    mock_span.attributes = {"some_attr": compiled_text}
    mock_span.context = Mock()
    mock_span.context.trace_id = 12345
    mock_span.context.span_id = 67890
    mock_span.name = "test-span"

    # Import and create span processor (we'll mock the exporter)
    from langfuse._client.span_processor import LangfuseSpanProcessor

    with patch("langfuse._client.span_processor.OTLPSpanExporter"):
        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,  # Disable server export for testing
        )

        # Call the method under test
        processor._emit_prompt_attributes(mock_span)

        # Check that prompt_for_span was emitted
        ctx = exec_ctx.get_current_context()
        assert ctx is not None

        update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

        assert update is not None
        assert update.status == "unchanged"
        assert update.prompt_for_span is not None
        assert isinstance(update.prompt_for_span, PromptForSpan)
        assert update.prompt_for_span.prompt_id == prompt.id
        assert update.prompt_for_span.version_id == "v0"  # default version
        assert update.prompt_for_span.trace_id == format_trace_id(12345)
        assert update.prompt_for_span.span_id == format_span_id(67890)
        assert update.prompt_for_span.variables is None  # No variables in this case

    # Clean up
    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_emit_prompt_attributes_with_variables():
    """Test prompt emission with variables."""

    # Initialize execution context
    run_id = "test-run-456"
    exec_ctx.init_run(run_id)

    # Create a prompt with variables
    prompt = BasePrompt(
        versions="Hello {name}, you have {count} items",
        variables_definition=PromptTestVars,
    )
    variables = PromptTestVars(name="Alice", count=5)
    compiled_text = prompt.compile(variables)

    # Mock span
    mock_span = Mock()
    mock_span.attributes = {"prompt": compiled_text}
    mock_span.context = Mock()
    mock_span.context.trace_id = 11111
    mock_span.context.span_id = 22222
    mock_span.name = "llm-call"

    from langfuse._client.span_processor import LangfuseSpanProcessor

    with patch("langfuse._client.span_processor.OTLPSpanExporter"):
        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        processor._emit_prompt_attributes(mock_span)

        ctx = exec_ctx.get_current_context()
        update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

        assert update.prompt_for_span is not None
        assert update.prompt_for_span.prompt_id == prompt.id
        assert update.prompt_for_span.variables == {"name": "Alice", "count": 5}

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_emit_prompt_attributes_no_match():
    """Test that no prompt emission occurs when span doesn't contain compiled prompts."""

    run_id = "test-run-789"
    exec_ctx.init_run(run_id)

    # Create a prompt but don't use it in span
    prompt = BasePrompt(versions="Some prompt")
    prompt.compile()  # Register it

    # Mock span with different text
    mock_span = Mock()
    mock_span.attributes = {"text": "Some other text not from a prompt"}
    mock_span.context = Mock()
    mock_span.name = "test-span"

    from langfuse._client.span_processor import LangfuseSpanProcessor

    with patch("langfuse._client.span_processor.OTLPSpanExporter"):
        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        processor._emit_prompt_attributes(mock_span)

        # Should not emit anything
        ctx = exec_ctx.get_current_context()
        # Try to get update with timeout - should timeout since no update was sent
        try:
            await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=0.1)
            assert False, "Should not have received any update"
        except asyncio.TimeoutError:
            pass  # Expected - no update should be sent

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_emit_prompt_attributes_no_attributes():
    """Test that no emission occurs when span has no attributes."""

    run_id = "test-run-no-attr"
    exec_ctx.init_run(run_id)

    mock_span = Mock()
    mock_span.attributes = None
    mock_span.context = None
    mock_span.name = "test-span"

    from langfuse._client.span_processor import LangfuseSpanProcessor

    with patch("langfuse._client.span_processor.OTLPSpanExporter"):
        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        # Should not crash
        processor._emit_prompt_attributes(mock_span)

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_emit_prompt_attributes_json_embedded():
    """Test prompt detection in JSON-embedded compiled prompts."""

    run_id = "test-run-json"
    exec_ctx.init_run(run_id)

    # Create prompt and compile
    prompt = BasePrompt(versions="Embedded prompt")
    compiled_text = prompt.compile()

    # Embed in JSON
    import json

    json_data = {"messages": [{"role": "user", "content": compiled_text}]}
    json_string = json.dumps(json_data)

    mock_span = Mock()
    mock_span.attributes = {"llm_input": json_string}  # Use proper JSON string
    mock_span.context = Mock()
    mock_span.context.trace_id = 33333
    mock_span.context.span_id = 44444
    mock_span.name = "llm-span"

    from langfuse._client.span_processor import LangfuseSpanProcessor

    with patch("langfuse._client.span_processor.OTLPSpanExporter"):
        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        processor._emit_prompt_attributes(mock_span)

        ctx = exec_ctx.get_current_context()
        update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

        assert update.prompt_for_span is not None
        assert update.prompt_for_span.prompt_id == prompt.id

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_on_start_emits_partial_trace():
    """Test that on_start emits partial trace data to execution context."""

    run_id = "test-run-start"
    exec_ctx.init_run(run_id)

    # Mock span for on_start (Span type)
    mock_span = Mock()
    mock_span.name = "test-span"
    mock_span.start_time = 1234567890000000000  # nanoseconds
    mock_span.context = Mock()
    mock_span.context.trace_id = 12345
    mock_span.context.span_id = 67890
    mock_span.parent = Mock()
    mock_span.parent.span_id = 11111
    mock_span.attributes = {"key": "value"}
    mock_span.kind = Mock()
    mock_span.kind.name = "INTERNAL"

    from langfuse._client.span_processor import LangfuseSpanProcessor

    with patch("langfuse._client.span_processor.OTLPSpanExporter"):
        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        # Call on_start
        processor.on_start(mock_span)

        # Check that trace was emitted
        ctx = exec_ctx.get_current_context()
        update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

        assert update is not None
        assert update.status == "unchanged"
        assert update.trace is not None
        assert update.trace["event"] == "span_start"
        assert update.trace["span_name"] == "test-span"
        assert update.trace["start_time_unix_nano"] == "1234567890000000000"
        assert update.trace["trace_id"] == format_trace_id(12345)
        assert update.trace["span_id"] == format_span_id(67890)
        assert update.trace["parent_span_id"] == format_span_id(11111)
        assert update.trace["attributes"] == {"key": "value"}
        assert update.trace["kind"] == "INTERNAL"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_on_end_emits_full_trace():
    """Test that on_end emits full trace data to execution context."""

    run_id = "test-run-end"
    exec_ctx.init_run(run_id)

    # Mock ReadableSpan for on_end
    mock_span = Mock()
    mock_span.name = "test-span"
    mock_span.instrumentation_scope = None  # To pass _is_langfuse_project_span check
    mock_span.parent = None  # Explicitly set parent to None
    mock_span.attributes = {}  # Empty attributes
    mock_span.context = None  # No context for prompt emission

    from langfuse._client.span_processor import LangfuseSpanProcessor

    # Mock the encode_spans and MessageToDict
    mock_proto = Mock()
    expected_trace_data = {"resourceSpans": []}

    with patch("langfuse._client.span_processor.OTLPSpanExporter"), patch(
        "langfuse._client.span_processor.encode_spans", return_value=mock_proto
    ), patch(
        "langfuse._client.span_processor.MessageToDict",
        return_value=expected_trace_data,
    ), patch(
        "langfuse._client.span_processor.span_formatter", return_value="mocked"
    ):

        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        # Call on_end
        processor.on_end(mock_span)

        # Check that trace was emitted
        ctx = exec_ctx.get_current_context()
        update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

        assert update is not None
        assert update.status == "unchanged"
        assert update.trace == expected_trace_data

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_on_start_and_on_end_emissions():
    """Test that both on_start and on_end emit trace data."""

    run_id = "test-run-both"
    exec_ctx.init_run(run_id)

    # Mock span for on_start
    mock_span_start = Mock()
    mock_span_start.name = "test-span"
    mock_span_start.start_time = 1234567890000000000
    mock_span_start.context = Mock()
    mock_span_start.context.trace_id = 12345
    mock_span_start.context.span_id = 67890
    mock_span_start.parent = None
    mock_span_start.attributes = {}
    mock_span_start.kind = Mock()
    mock_span_start.kind.name = "INTERNAL"

    # Mock ReadableSpan for on_end (same span conceptually)
    mock_span_end = Mock()
    mock_span_end.name = "test-span"
    mock_span_end.instrumentation_scope = None
    mock_span_end.attributes = {}
    mock_span_end.context = None

    from langfuse._client.span_processor import LangfuseSpanProcessor

    expected_trace_data = {"resourceSpans": []}

    with patch("langfuse._client.span_processor.OTLPSpanExporter"), patch(
        "langfuse._client.span_processor.encode_spans", return_value=Mock()
    ), patch(
        "langfuse._client.span_processor.MessageToDict",
        return_value=expected_trace_data,
    ), patch(
        "langfuse._client.span_processor.span_formatter", return_value="mocked"
    ):

        processor = LangfuseSpanProcessor(
            public_key="test",
            secret_key="test",
            base_url="http://test.com",
            server_export_enabled=False,
        )

        # Call on_start
        processor.on_start(mock_span_start)

        # Call on_end
        processor.on_end(mock_span_end)

        # Check both emissions
        ctx = exec_ctx.get_current_context()

        # First update should be from on_start
        update1 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
        assert update1.trace["event"] == "span_start"

        # Second update should be from on_end
        update2 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
        assert update2.trace == expected_trace_data

    exec_ctx.unregister_run(run_id)
