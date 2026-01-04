"""Test pause timing adjustment to ensure pause duration is not counted in spans."""

import threading
import time
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.resources import Resource

from pixie import execution_context as exec_ctx
from pixie.types import BreakpointConfig


@pytest.fixture(autouse=True)
def reset_execution_context() -> Generator[None, None, None]:
    """Reset execution context state between tests."""
    yield
    # pylint: disable=protected-access
    exec_ctx._execution_context.set(None)  # type: ignore[attr-defined]
    exec_ctx._active_runs.clear()  # type: ignore[attr-defined]
    # pylint: enable=protected-access


class MockSpanExporter(SpanExporter):
    """Mock exporter to capture exported spans."""

    def __init__(self):
        self.exported_spans = []

    def export(self, spans):
        self.exported_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def test_pause_before_span_adjusts_timing():
    """Test that pausing BEFORE a span starts adjusts the span's start time."""
    # Initialize execution context
    run_id = "test-pause-before"
    ctx = exec_ctx.init_run(run_id)

    # Import the processor
    from langfuse._client.span_processor import LangfuseSpanProcessor

    # Create a tracer with a simple exporter (not LangfuseSpanProcessor)
    tracer_provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    mock_exporter = MockSpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(mock_exporter))
    tracer = tracer_provider.get_tracer(
        "langfuse", attributes={"public_key": "pk-test"}
    )

    # Mock the OTLPSpanExporter to avoid network calls
    with patch("langfuse._client.span_processor.OTLPSpanExporter") as MockExporter:
        mock_exporter_instance = Mock()
        mock_exporter_instance.export = Mock(return_value=SpanExportResult.SUCCESS)
        MockExporter.return_value = mock_exporter_instance

        # Create the processor instance
        processor_instance = LangfuseSpanProcessor(
            public_key="pk-test",
            secret_key="sk-test",
            base_url="https://test.langfuse.com",
        )

        # Create a thread that will resume after a short pause
        pause_duration_seconds = 0.5

        def resume_after_delay():
            time.sleep(pause_duration_seconds)
            exec_ctx.resume_run(run_id)

        resume_thread = threading.Thread(target=resume_after_delay)

        # Patch the exec_ctx module in span_processor
        with patch("langfuse._client.span_processor.exec_ctx") as mock_exec_ctx:
            # Setup mock to return our context and config
            mock_exec_ctx.get_current_breakpoint_config.return_value = BreakpointConfig(
                id="bp-1", timing="BEFORE", breakpoint_types=["LLM"]
            )
            mock_exec_ctx.get_current_context.return_value = ctx
            mock_exec_ctx.emit_status_update_sync = Mock()
            mock_exec_ctx.wait_for_resume = ctx.resume_event.wait

            # Start resume thread
            resume_thread.start()

            # Record time before span creation
            time_before_span = time.time_ns()

            # Create a span with LLM-like name to trigger breakpoint
            span = tracer.start_span("openai.chat.completion")  # type: ignore[misc]
            try:
                # Manually call the processor's on_start to test pause logic
                processor_instance.on_start(span)  # type: ignore[arg-type]

                # The pause should have occurred, and start_time should be adjusted
                time_after_resume = time.time_ns()

                # Get the span's start time
                # pylint: disable=protected-access
                span_start_time = span._start_time  # type: ignore[attr-defined]
                # pylint: enable=protected-access

                # Verify that the span start time was adjusted
                # It should be later than time_before_span by roughly pause_duration
                assert span_start_time > time_before_span
                assert (
                    span_start_time >= time_after_resume - 100_000_000
                )  # 100ms tolerance

                # Verify pause attributes were added
                assert span.attributes is not None  # type: ignore[attr-defined]
                assert "pixie.pause.duration_ns" in span.attributes  # type: ignore[attr-defined]
                assert "pixie.pause.adjusted" in span.attributes  # type: ignore[attr-defined]
                assert span.attributes["pixie.pause.adjusted"] is True  # type: ignore[attr-defined]

                # Verify pause duration is approximately correct (within 100ms tolerance)
                recorded_pause_ns = span.attributes["pixie.pause.duration_ns"]  # type: ignore[attr-defined]
                expected_pause_ns = pause_duration_seconds * 1e9
                assert abs(recorded_pause_ns - expected_pause_ns) < 100_000_000
            finally:
                span.end()

            resume_thread.join(timeout=2.0)

    # Cleanup
    exec_ctx.unregister_run(run_id)
    tracer_provider.shutdown()
