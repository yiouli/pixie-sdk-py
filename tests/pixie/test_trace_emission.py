"""Test trace emission to GraphQL subscription."""

import asyncio
import pytest

import pixie.execution_context as exec_ctx


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


if __name__ == "__main__":
    asyncio.run(test_trace_emission_to_execution_context())
