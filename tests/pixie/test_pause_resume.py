"""Test pause/resume functionality."""

import asyncio
import pytest
from pixie.types import (
    BreakpointConfig,
    BreakpointDetail,
)
from pixie import execution_context as exec_ctx


@pytest.fixture(autouse=True)
def reset_execution_context():
    """Reset execution context state between tests."""
    yield
    exec_ctx._execution_context.set(None)
    exec_ctx._active_runs.clear()


def test_pause_config_creation():
    """Test creating pause config."""
    config = BreakpointConfig(
        id="bp-1",
        timing="BEFORE",
        breakpoint_types=["LLM", "TOOL"],
    )
    assert config.timing == "BEFORE"
    assert len(config.breakpoint_types) == 2


@pytest.mark.asyncio
async def test_execution_context_management():
    """Test execution context registration and retrieval."""
    run_id = "test-run-123"

    ctx = exec_ctx.init_run(run_id)
    assert ctx.run_id == run_id
    assert isinstance(ctx.status_queue, asyncio.Queue)

    retrieved_ctx = exec_ctx.get_run_context(run_id)
    assert retrieved_ctx is not None
    assert retrieved_ctx.run_id == run_id
    assert exec_ctx.get_current_breakpoint_config() is None

    exec_ctx.unregister_run(run_id)
    assert exec_ctx.get_run_context(run_id) is None


@pytest.mark.asyncio
async def test_pause_config_setting():
    """Test setting pause config for a run."""
    run_id = "test-run-456"
    exec_ctx.init_run(run_id)

    exec_ctx.set_breakpoint(run_id, timing="AFTER", types=["LLM"])

    retrieved_ctx = exec_ctx.get_run_context(run_id)
    assert retrieved_ctx is not None
    assert retrieved_ctx.breakpoint_config is not None
    assert retrieved_ctx.breakpoint_config.timing == "AFTER"
    assert retrieved_ctx.breakpoint_config.breakpoint_types == ["LLM"]

    # Simulate resume to clear breakpoint config via wait_for_resume
    retrieved_ctx.resume_event.set()
    exec_ctx.wait_for_resume()
    assert retrieved_ctx.breakpoint_config is None

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_pause_and_resume():
    """Test pause and resume flow."""
    run_id = "test-run-789"
    ctx = exec_ctx.init_run(run_id)
    assert ctx is not None

    # Initially not resumed
    assert not ctx.resume_event.is_set()

    # Resume
    success = exec_ctx.resume_run(run_id)
    assert success
    assert ctx.resume_event.is_set()

    # Second resume is a no-op
    assert not exec_ctx.resume_run(run_id)

    # Emit an explicit status update to the queue
    await exec_ctx.emit_status_update(status="running")
    update = await asyncio.wait_for(ctx.status_queue.get(), timeout=1.0)
    assert update is not None
    assert update.status == "running"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_status_update_emission():
    """Test emitting status updates to queue."""
    run_id = "test-run-abc"
    ctx = exec_ctx.init_run(run_id)

    await exec_ctx.emit_status_update(
        status="paused",
        data='{"span_type": "LLM", "span_name": "openai.chat.completions"}',
    )

    received_update = await asyncio.wait_for(ctx.status_queue.get(), timeout=1.0)
    assert received_update is not None
    assert received_update.status == "paused"
    import json

    assert received_update.data is not None
    data = json.loads(received_update.data)
    assert data["span_type"] == "LLM"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_invalid_run_id():
    """Test operations with invalid run ID."""
    # Try to set pause config for non-existent run
    with pytest.raises(ValueError):
        exec_ctx.set_breakpoint("non-existent-run", timing="BEFORE", types=["TOOL"])

    # Try to resume non-existent run
    with pytest.raises(ValueError):
        exec_ctx.resume_run("non-existent-run")

    # Check if non-existent run is active
    assert exec_ctx.get_run_context("non-existent-run") is None


@pytest.mark.asyncio
async def test_breakpoint_data():
    """Test breakpoint data creation."""
    bp = BreakpointDetail(
        span_name="openai.chat.completions.analyze",
        breakpoint_type="LLM",
        breakpoint_timing="BEFORE",
        span_attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "gpt-4",
        },
    )

    assert bp.span_name == "openai.chat.completions.analyze"
    assert bp.breakpoint_type == "LLM"
    assert bp.breakpoint_timing == "BEFORE"
    assert (
        bp.span_attributes is not None
        and bp.span_attributes.get("gen_ai.request.model") == "gpt-4"
    )


@pytest.mark.asyncio
async def test_status_update_with_breakpoint():
    """Test status update emission with breakpoint data."""
    run_id = "test-run-breakpoint"
    ctx = exec_ctx.init_run(run_id)

    # Create breakpoint data
    bp = BreakpointDetail(
        span_name="tool.search_knowledge_base",
        breakpoint_type="TOOL",
        breakpoint_timing="AFTER",
        span_attributes={"tool.name": "search_knowledge_base"},
    )

    # Emit status update with breakpoint
    await exec_ctx.emit_status_update(
        status="paused",
        data='{"span_type": "TOOL", "span_name": "tool.search_knowledge_base"}',
        breakpt=bp,
    )

    # Verify update was queued
    received_update = await asyncio.wait_for(ctx.status_queue.get(), timeout=1.0)
    assert received_update is not None
    assert received_update.status == "paused"
    import json

    assert received_update.data is not None
    data = json.loads(received_update.data)
    assert data["span_type"] == "TOOL"
    assert received_update.breakpoint is not None
    assert received_update.breakpoint.breakpoint_timing == "AFTER"
    assert received_update.breakpoint.span_attributes is not None
    assert (
        received_update.breakpoint.span_attributes["tool.name"]
        == "search_knowledge_base"
    )

    exec_ctx.unregister_run(run_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
