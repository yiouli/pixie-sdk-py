"""Test trace emission to GraphQL subscription."""

import asyncio
import pytest

import pixie.execution_context as exec_ctx
from pixie.types import PromptForSpan


@pytest.mark.asyncio
async def test_trace_emission_to_execution_context():
    """Test that execution context can receive trace data."""

    # Initialize execution context
    run_id = "test-run-123"
    exec_ctx.init_run(run_id)

    # Emit a test trace using the sync helper
    test_trace = {
        "event": "span_start",
        "span_name": "test-span",
        "trace_id": "test-trace-id",
    }

    exec_ctx.emit_status_update(
        status="running",
        trace=test_trace,
    )

    # Get the update from the queue
    ctx = exec_ctx.get_current_context()
    assert ctx is not None

    update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

    # Verify the trace was received
    assert update is not None
    assert update.status == "running"
    assert update.trace is not None
    assert update.trace["event"] == "span_start"
    assert update.trace["span_name"] == "test-span"

    # Clean up
    exec_ctx.unregister_run(run_id)

    print("Test passed - trace emission mechanism works correctly")


@pytest.mark.asyncio
async def test_prompt_for_span_emission():
    """Test that PromptForSpan data can be emitted via execution context."""

    # Initialize execution context
    run_id = "test-run-prompt-123"
    exec_ctx.init_run(run_id)

    # Create a PromptForSpan instance
    prompt_info = PromptForSpan(
        trace_id=12345,
        span_id=67890,
        prompt_id="test-prompt-id",
        version_id="v1",
        variables={"param": "value"},
    )

    # Emit the prompt info
    exec_ctx.emit_status_update(
        status="running",
        prompt_for_span=prompt_info,
    )

    # Get the update from the queue
    ctx = exec_ctx.get_current_context()
    assert ctx is not None

    update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

    # Verify the prompt info was received
    assert update is not None
    assert update.status == "running"
    assert update.prompt_for_span is not None
    assert update.prompt_for_span.trace_id == 12345
    assert update.prompt_for_span.span_id == 67890
    assert update.prompt_for_span.prompt_id == "test-prompt-id"
    assert update.prompt_for_span.version_id == "v1"
    assert update.prompt_for_span.variables == {"param": "value"}

    # Clean up
    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_combined_trace_and_prompt_emission():
    """Test emitting both trace and prompt data in the same update."""

    # Initialize execution context
    run_id = "test-run-combined-456"
    exec_ctx.init_run(run_id)

    # Create test data
    test_trace = {
        "event": "span_start",
        "span_name": "llm-span",
        "trace_id": "test-trace-id",
    }

    prompt_info = PromptForSpan(
        trace_id=11111,
        span_id=22222,
        prompt_id="combined-prompt-id",
        version_id="v2",
        variables=None,
    )

    # Emit both in one update
    exec_ctx.emit_status_update(
        status="running",
        trace=test_trace,
        prompt_for_span=prompt_info,
    )

    # Get the update
    ctx = exec_ctx.get_current_context()
    assert ctx is not None

    update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)

    # Verify both pieces of data
    assert update is not None
    assert update.status == "running"
    assert update.trace is not None
    assert update.trace["span_name"] == "llm-span"
    assert update.prompt_for_span is not None
    assert update.prompt_for_span.prompt_id == "combined-prompt-id"
    assert update.prompt_for_span.variables is None

    # Clean up
    exec_ctx.unregister_run(run_id)


if __name__ == "__main__":
    asyncio.run(test_trace_emission_to_execution_context())
    asyncio.run(test_prompt_for_span_emission())
    asyncio.run(test_combined_trace_and_prompt_emission())
