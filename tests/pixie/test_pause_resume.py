"""Test pause/resume functionality."""

import asyncio
import pytest
from pixie.types import (
    PauseConfig,
    ExecutionContext,
    AppRunUpdate,
    Breakpoint,
)
from pixie import execution_context as exec_ctx


def test_pause_config_creation():
    """Test creating pause config."""
    config = PauseConfig(
        mode="BEFORE",
        pausible_points=["LLM", "TOOL"],
    )
    assert config.mode == "BEFORE"
    assert len(config.pausible_points) == 2


@pytest.mark.asyncio
async def test_execution_context_management():
    """Test execution context registration and retrieval."""
    run_id = "test-run-123"
    queue = asyncio.Queue()
    resume_event = asyncio.Event()

    ctx = ExecutionContext(
        run_id=run_id,
        status_queue=queue,
        resume_event=resume_event,
    )

    # Register run
    exec_ctx.register_run(ctx)
    assert exec_ctx.is_run_active(run_id)

    # Retrieve context
    retrieved_ctx = exec_ctx.get_run_context(run_id)
    assert retrieved_ctx is not None
    assert retrieved_ctx.run_id == run_id

    # Unregister run
    exec_ctx.unregister_run(run_id)
    assert not exec_ctx.is_run_active(run_id)


@pytest.mark.asyncio
async def test_pause_config_setting():
    """Test setting pause config for a run."""
    run_id = "test-run-456"
    ctx = ExecutionContext(run_id=run_id)
    exec_ctx.register_run(ctx)

    config = PauseConfig(
        mode="AFTER",
        pausible_points=["LLM"],
    )

    # Set pause config
    success = exec_ctx.set_pause_config(run_id, config)
    assert success

    # Verify config was set
    retrieved_ctx = exec_ctx.get_run_context(run_id)
    assert retrieved_ctx is not None
    assert retrieved_ctx.pause_config is not None
    assert retrieved_ctx.pause_config.mode == "AFTER"

    # Clear pause config
    exec_ctx.clear_pause_config(run_id)
    retrieved_ctx = exec_ctx.get_run_context(run_id)
    assert retrieved_ctx is not None
    assert retrieved_ctx.pause_config is None

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_pause_and_resume():
    """Test pause and resume flow."""
    run_id = "test-run-789"
    queue = asyncio.Queue()
    resume_event = asyncio.Event()
    resume_event.set()  # Start resumed

    ctx = ExecutionContext(
        run_id=run_id,
        status_queue=queue,
        resume_event=resume_event,
    )
    exec_ctx.register_run(ctx)

    # Initially not paused
    assert not exec_ctx.is_run_paused(run_id)

    # Simulate pause by clearing event
    resume_event.clear()
    assert exec_ctx.is_run_paused(run_id)

    # Resume
    success = exec_ctx.trigger_resume(run_id)
    assert success
    assert not exec_ctx.is_run_paused(run_id)

    # Verify status update was queued
    update = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert update.status == "resumed"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_status_update_emission():
    """Test emitting status updates to queue."""
    run_id = "test-run-abc"
    queue = asyncio.Queue()

    ctx = ExecutionContext(
        run_id=run_id,
        status_queue=queue,
    )

    # Set context
    exec_ctx.set_execution_context(ctx)
    exec_ctx.register_run(ctx)

    # Emit status update
    update = AppRunUpdate(
        run_id=run_id,
        status="paused",
        data='{"span_type": "LLM", "span_name": "openai.chat.completions"}',
    )
    await exec_ctx.emit_status_update(update)

    # Verify update was queued
    received_update = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert received_update.status == "paused"
    import json

    data = json.loads(received_update.data)
    assert data["span_type"] == "LLM"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_invalid_run_id():
    """Test operations with invalid run ID."""
    # Try to set pause config for non-existent run
    config = PauseConfig(
        mode="BEFORE",
        pausible_points=["TOOL"],
    )
    success = exec_ctx.set_pause_config("non-existent-run", config)
    assert not success

    # Try to resume non-existent run
    success = exec_ctx.trigger_resume("non-existent-run")
    assert not success

    # Check if non-existent run is active
    assert not exec_ctx.is_run_active("non-existent-run")


@pytest.mark.asyncio
async def test_breakpoint_data():
    """Test breakpoint data creation."""
    bp = Breakpoint(
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
    queue = asyncio.Queue()

    ctx = ExecutionContext(
        run_id=run_id,
        status_queue=queue,
    )

    # Set context
    exec_ctx.set_execution_context(ctx)
    exec_ctx.register_run(ctx)

    # Create breakpoint data
    bp = Breakpoint(
        span_name="tool.search_knowledge_base",
        breakpoint_type="TOOL",
        breakpoint_timing="AFTER",
        span_attributes={"tool.name": "search_knowledge_base"},
    )

    # Emit status update with breakpoint
    update = AppRunUpdate(
        run_id=run_id,
        status="paused",
        data='{"span_type": "TOOL", "span_name": "tool.search_knowledge_base"}',
        breakpoint=bp,
    )
    await exec_ctx.emit_status_update(update)

    # Verify update was queued
    received_update = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert received_update.status == "paused"
    import json

    data = json.loads(received_update.data)
    assert data["span_type"] == "TOOL"
    assert received_update.breakpoint is not None
    assert received_update.breakpoint.breakpoint_timing == "AFTER"
    assert (
        received_update.breakpoint.span_attributes["tool.name"]
        == "search_knowledge_base"
    )

    exec_ctx.unregister_run(run_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
