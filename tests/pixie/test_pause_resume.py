"""Test pause/resume functionality."""

import asyncio
import json
from typing import Generator, cast
import pytest
from pixie.types import (
    BreakpointConfig,
    BreakpointDetail,
)
from pixie import execution_context as exec_ctx


@pytest.fixture(autouse=True)
def reset_execution_context() -> Generator[None, None, None]:
    """Reset execution context state between tests."""
    yield
    exec_ctx._execution_context.set(None)
    exec_ctx._active_runs.clear()


def test_pause_config_creation() -> None:
    """Test creating pause config."""
    config = BreakpointConfig(
        id="bp-1",
        timing="BEFORE",
        breakpoint_types=["LLM", "TOOL"],
    )
    assert config.timing == "BEFORE"
    assert len(config.breakpoint_types) == 2


@pytest.mark.asyncio
async def test_execution_context_management() -> None:
    """Test execution context registration and retrieval."""
    run_id = "test-run-123"

    ctx = exec_ctx.init_run(run_id)
    assert ctx.run_id == run_id

    retrieved_ctx = exec_ctx.get_run_context(run_id)
    assert retrieved_ctx is not None
    assert retrieved_ctx.run_id == run_id
    assert exec_ctx.get_current_breakpoint_config() is None

    exec_ctx.unregister_run(run_id)
    assert exec_ctx.get_run_context(run_id) is None


@pytest.mark.asyncio
async def test_pause_config_setting() -> None:
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
    assert retrieved_ctx.breakpoint_config is None  # type: ignore[unreachable]

    exec_ctx.unregister_run(run_id)  # type: ignore[unreachable]


@pytest.mark.asyncio
async def test_pause_and_resume() -> None:
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
    exec_ctx.emit_status_update(status="running")
    update = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update is not None
    assert update.status == "running"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_status_update_emission() -> None:
    """Test emitting status updates to queue."""
    run_id = "test-run-abc"
    ctx = exec_ctx.init_run(run_id)

    exec_ctx.emit_status_update(
        status="paused",
        data='{"span_type": "LLM", "span_name": "openai.chat.completions"}',
    )

    received_update = await asyncio.wait_for(
        ctx.status_queue.async_q.get(), timeout=1.0
    )
    assert received_update is not None
    assert received_update.status == "paused"

    assert received_update.data is not None
    data = json.loads(cast(str, received_update.data))
    assert data["span_type"] == "LLM"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_invalid_run_id() -> None:
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
async def test_breakpoint_data() -> None:
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
async def test_status_update_with_breakpoint() -> None:
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
    exec_ctx.emit_status_update(
        status="paused",
        data='{"span_type": "TOOL", "span_name": "tool.search_knowledge_base"}',
        breakpt=bp,
    )

    # Verify update was queued
    received_update = await asyncio.wait_for(
        ctx.status_queue.async_q.get(), timeout=1.0
    )
    assert received_update is not None
    assert received_update.status == "paused"

    assert received_update.data is not None
    data = json.loads(cast(str, received_update.data))
    assert data["span_type"] == "TOOL"
    assert received_update.breakpoint is not None
    assert received_update.breakpoint.breakpoint_timing == "AFTER"
    assert received_update.breakpoint.span_attributes is not None
    assert (
        received_update.breakpoint.span_attributes["tool.name"]
        == "search_knowledge_base"
    )

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_reload_run_context() -> None:
    """Test reloading execution context from global registry."""
    run_id = "test-run-reload"
    exec_ctx.init_run(run_id)

    # Set a breakpoint config
    exec_ctx.set_breakpoint(run_id, timing="BEFORE", types=["LLM"])

    # Clear the context var to simulate switching contexts
    exec_ctx._execution_context.set(None)
    assert exec_ctx._execution_context.get() is None

    # Reload from registry
    exec_ctx.reload_run_context(run_id)

    # Verify context was restored
    reloaded_ctx = exec_ctx._execution_context.get()
    assert reloaded_ctx is not None
    assert reloaded_ctx.run_id == run_id
    assert reloaded_ctx.breakpoint_config is not None
    assert reloaded_ctx.breakpoint_config.timing == "BEFORE"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_reload_run_context_invalid_run_id() -> None:
    """Test reloading execution context with invalid run ID."""
    with pytest.raises(ValueError, match="Run ID 'invalid-run' not found"):
        exec_ctx.reload_run_context("invalid-run")


@pytest.mark.asyncio
async def test_cancel_run() -> None:
    """Test cancelling a running or paused run."""
    run_id = "test-run-cancel"
    ctx = exec_ctx.init_run(run_id)

    # Initially not cancelled
    assert not ctx.cancelled

    # Cancel the run
    success = exec_ctx.cancel_run()
    assert success
    assert ctx.cancelled
    assert ctx.resume_event.is_set()  # Should trigger resume

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_cancel_run_already_cancelled() -> None:
    """Test cancelling an already cancelled run."""
    run_id = "test-run-cancel-twice"
    ctx = exec_ctx.init_run(run_id)

    # First cancellation
    success = exec_ctx.cancel_run()
    assert success
    assert ctx.cancelled

    # Second cancellation should return False
    success = exec_ctx.cancel_run()
    assert not success
    assert ctx.cancelled

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_cancel_run_no_context() -> None:
    """Test cancelling when no execution context is set."""
    # Clear context var
    exec_ctx._execution_context.set(None)

    # Should return False and log warning
    success = exec_ctx.cancel_run()
    assert not success


@pytest.mark.asyncio
async def test_wait_for_resume_with_cancellation() -> None:
    """Test that wait_for_resume raises exception when run is cancelled."""
    run_id = "test-run-cancel-during-pause"
    ctx = exec_ctx.init_run(run_id)

    # Set up a background task that will cancel the run
    async def cancel_after_delay() -> None:
        await asyncio.sleep(0.1)
        ctx.cancelled = True
        ctx.resume_event.set()

    # Start the cancellation task
    cancel_task = asyncio.create_task(cancel_after_delay())

    # Wait for resume should raise AppRunCancelled
    from pixie.types import AppRunCancelled

    with pytest.raises(AppRunCancelled, match=f"Run {run_id} was cancelled"):
        # Run wait_for_resume in a thread since it blocks
        import concurrent.futures

        def wait_with_context() -> None:
            # Reload context in this thread since ContextVar is thread-local
            exec_ctx.reload_run_context(run_id)
            exec_ctx.wait_for_resume()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(wait_with_context)
            # Give cancellation task time to run
            await cancel_task
            # This should raise AppRunCancelled
            future.result(timeout=2.0)

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_multiple_status_updates() -> None:
    """Test emitting multiple status updates to queue."""
    run_id = "test-run-multi-status"
    ctx = exec_ctx.init_run(run_id)

    # Emit multiple updates
    exec_ctx.emit_status_update(status="running")
    exec_ctx.emit_status_update(status="paused", data='{"step": 1}')
    exec_ctx.emit_status_update(status="running")
    exec_ctx.emit_status_update(status="completed", data='{"result": "success"}')

    # Verify all updates are in queue in order
    update1 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update1 is not None
    assert update1.status == "running"

    update2 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update2 is not None
    assert update2.status == "paused"

    update3 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update3 is not None
    assert update3.status == "running"

    update4 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update4 is not None
    assert update4.status == "completed"

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_terminal_status_update() -> None:
    """Test emitting terminal status update (None)."""
    run_id = "test-run-terminal"
    ctx = exec_ctx.init_run(run_id)

    # Emit a normal update first
    exec_ctx.emit_status_update(status="running")

    # Emit terminal update (None)
    exec_ctx.emit_status_update(status=None)

    # Verify both are in queue
    update1 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update1 is not None
    assert update1.status == "running"

    update2 = await asyncio.wait_for(ctx.status_queue.async_q.get(), timeout=1.0)
    assert update2 is None  # Terminal sentinel

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_init_run_already_set() -> None:
    """Test initializing run when context is already set."""
    run_id = "test-run-first"
    exec_ctx.init_run(run_id)

    # Try to init another run without clearing context
    with pytest.raises(RuntimeError, match="Execution context is already set"):
        exec_ctx.init_run("test-run-second")

    exec_ctx.unregister_run(run_id)


@pytest.mark.asyncio
async def test_wait_for_resume_no_context() -> None:
    """Test wait_for_resume when no execution context is set."""
    # Clear context var
    exec_ctx._execution_context.set(None)

    # Should return without blocking or raising
    exec_ctx.wait_for_resume()  # Should just log warning and return


@pytest.mark.asyncio
async def test_emit_status_update_no_context() -> None:
    """Test emitting status update when no execution context is set."""
    # Clear context var
    exec_ctx._execution_context.set(None)

    # Should not raise an error, just do nothing
    exec_ctx.emit_status_update(status="running")


@pytest.mark.asyncio
async def test_pause_resume_full_cycle() -> None:
    """Test full pause/resume cycle with breakpoint config."""
    run_id = "test-run-full-cycle"
    ctx = exec_ctx.init_run(run_id)

    # Set breakpoint
    config = exec_ctx.set_breakpoint(run_id, timing="BEFORE", types=["LLM", "TOOL"])
    assert config is not None
    assert ctx.breakpoint_config is not None

    # Create breakpoint detail
    bp = BreakpointDetail(
        span_name="openai.chat.completions",
        breakpoint_type="LLM",
        breakpoint_timing="BEFORE",
        span_attributes={"gen_ai.request.model": "gpt-4"},
    )

    # Emit paused status
    exec_ctx.emit_status_update(status="paused", breakpt=bp)

    # Resume in background thread
    import concurrent.futures

    def wait_for_resume_with_context() -> None:
        # Reload context in this thread since ContextVar is thread-local
        exec_ctx.reload_run_context(run_id)
        exec_ctx.wait_for_resume()

    def resume_after_delay() -> None:
        import time

        time.sleep(0.1)
        exec_ctx.resume_run(run_id)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        resume_future = executor.submit(resume_after_delay)

        # Wait for resume (blocks) - must reload context first
        wait_future = executor.submit(wait_for_resume_with_context)
        wait_future.result(timeout=2.0)

        resume_future.result(timeout=1.0)

    # Verify breakpoint config was cleared
    # Need to get fresh reference since context was modified in another thread
    fresh_ctx = exec_ctx.get_run_context(run_id)
    assert fresh_ctx is not None
    assert fresh_ctx.breakpoint_config is None
    assert not fresh_ctx.resume_event.is_set()

    # Verify status update was queued
    update = await asyncio.wait_for(fresh_ctx.status_queue.async_q.get(), timeout=1.0)
    assert update is not None
    assert update.status == "paused"
    assert update.breakpoint is not None

    exec_ctx.unregister_run(run_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
