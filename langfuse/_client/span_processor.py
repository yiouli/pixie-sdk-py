"""Span processor for Langfuse OpenTelemetry integration.

This module defines the LangfuseSpanProcessor class, which extends OpenTelemetry's
BatchSpanProcessor with Langfuse-specific functionality. It handles exporting
spans to the Langfuse API with proper authentication and filtering.

Key features:
- HTTP-based span export to Langfuse API
- Basic authentication with Langfuse API keys
- Configurable batch processing behavior
- Project-scoped span filtering to prevent cross-project data leakage
- Pause/resume support for execution control
"""

import base64
import os
import time
from typing import Dict, List, Optional

from google.protobuf.json_format import MessageToDict
from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id, SpanKind

from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.environment_variables import (
    LANGFUSE_FLUSH_AT,
    LANGFUSE_FLUSH_INTERVAL,
    LANGFUSE_OTEL_TRACES_EXPORT_PATH,
)
from langfuse._client.propagation import _get_propagated_attributes_from_context
from langfuse._client.utils import span_formatter
from langfuse.logger import langfuse_logger
from langfuse.version import __version__ as langfuse_version


from pixie import execution_context as exec_ctx
from pixie.prompts.prompt import get_compiled_prompt
from pixie.types import BreakpointDetail, BreakpointType, PromptForSpan


class LangfuseSpanProcessor(BatchSpanProcessor):
    """OpenTelemetry span processor that exports spans to the Langfuse API.

    This processor extends OpenTelemetry's BatchSpanProcessor with Langfuse-specific functionality:
    1. Project-scoped span filtering to prevent cross-project data leakage
    2. Instrumentation scope filtering to block spans from specific libraries/frameworks
    3. Configurable batch processing parameters for optimal performance
    4. HTTP-based span export to the Langfuse OTLP endpoint (optional, can be disabled)
    5. Debug logging for span processing operations
    6. Authentication with Langfuse API using Basic Auth (when server export enabled)
    7. Pixie integration for pause/resume and trace emission to GraphQL subscriptions

    The processor is designed to efficiently handle large volumes of spans with
    minimal overhead, while ensuring spans are only sent to the correct project.
    It integrates with OpenTelemetry's standard span lifecycle, adding Langfuse-specific
    filtering and export capabilities.

    **Pixie-only Mode**: When initialized without credentials (server_export_enabled=False),
    the processor operates in Pixie-only mode where:
    - Spans are NOT exported to the Langfuse server
    - All Pixie features remain active: pause/resume, trace emission to GraphQL subscriptions
    - This enables using Pixie for observability without requiring a Langfuse account
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        server_export_enabled: bool = True,
    ):
        self.public_key = public_key
        self.server_export_enabled = server_export_enabled
        self.blocked_instrumentation_scopes = (
            blocked_instrumentation_scopes
            if blocked_instrumentation_scopes is not None
            else []
        )

        env_flush_at = os.environ.get(LANGFUSE_FLUSH_AT, None)
        flush_at = flush_at or int(env_flush_at) if env_flush_at is not None else None

        env_flush_interval = os.environ.get(LANGFUSE_FLUSH_INTERVAL, None)
        flush_interval = (
            flush_interval or float(env_flush_interval)
            if env_flush_interval is not None
            else None
        )

        # Only create real exporter if server export is enabled
        if server_export_enabled:
            basic_auth_header = "Basic " + base64.b64encode(
                f"{public_key}:{secret_key}".encode("utf-8")
            ).decode("ascii")

            # Prepare default headers
            default_headers = {
                "Authorization": basic_auth_header,
                "x-langfuse-sdk-name": "python",
                "x-langfuse-sdk-version": langfuse_version,
                "x-langfuse-public-key": public_key,
            }

            # Merge additional headers if provided
            headers = {**default_headers, **(additional_headers or {})}

            traces_export_path = os.environ.get(LANGFUSE_OTEL_TRACES_EXPORT_PATH, None)

            endpoint = (
                f"{base_url}/{traces_export_path}"
                if traces_export_path
                else f"{base_url}/api/public/otel/v1/traces"
            )

            langfuse_span_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers,
                timeout=timeout,
            )

            langfuse_logger.info(
                "Langfuse span processor initialized with server export enabled to %s",
                endpoint,
            )
        else:
            # Use NoOp exporter that doesn't send data anywhere
            from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

            class NoOpSpanExporter(SpanExporter):
                """No-op span exporter that doesn't export spans to any backend."""

                def export(self, spans):
                    # Don't export spans
                    return SpanExportResult.SUCCESS

                def shutdown(self):
                    pass

            langfuse_span_exporter = NoOpSpanExporter()

            langfuse_logger.info(
                "Langfuse span processor initialized in Pixie-only mode. "
                "Server export to Langfuse API is disabled. "
                "Pixie features (pause/resume, trace emission) remain active."
            )

        super().__init__(
            span_exporter=langfuse_span_exporter,
            export_timeout_millis=timeout * 1_000 if timeout else None,
            max_export_batch_size=flush_at,
            schedule_delay_millis=(
                flush_interval * 1_000 if flush_interval is not None else None
            ),
        )

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        context = parent_context or context_api.get_current()
        propagated_attributes = _get_propagated_attributes_from_context(context)

        if propagated_attributes:
            span.set_attributes(propagated_attributes)

            if span.context is not None:
                span_id = format_span_id(span.context.span_id)
                langfuse_logger.debug(
                    "Propagated %d attributes to span '%s': %s",
                    len(propagated_attributes),
                    span_id,
                    propagated_attributes,
                )
            else:
                langfuse_logger.debug(
                    "Propagated %d attributes to span with no valid context: %s",
                    len(propagated_attributes),
                    propagated_attributes,
                )

        # Check for pause before span starts (BEFORE_NEXT mode)
        self._check_breakpoint(span, is_before=True)

        # Emit partial trace at span start
        self._emit_trace_to_execution_context(span, is_start=True)

        return super().on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        # Only export spans that belong to the scoped project
        # This is important to not send spans to wrong project in multi-project setups
        if self._is_langfuse_span(span) and not self._is_langfuse_project_span(span):
            public_key_on_span = (
                span.instrumentation_scope.attributes.get("public_key")
                if span.instrumentation_scope and span.instrumentation_scope.attributes
                else None
            )
            langfuse_logger.debug(
                "Security: Span rejected - belongs to project '%s' but processor is for '%s'. "
                "This prevents cross-project data leakage in multi-project environments.",
                public_key_on_span,
                self.public_key,
            )
            return

        # Do not export spans from blocked instrumentation scopes
        if self._is_blocked_instrumentation_scope(span):
            return

        langfuse_logger.debug(
            "Trace: Processing span name='%s' | Full details:\n%s",
            getattr(span, "name", "unknown"),
            span_formatter(span),
        )

        # Check for pause after span ends (AFTER_NEXT mode)
        self._check_breakpoint(span, is_before=False)

        # Emit prompt attributes after breakpoint check
        self._emit_prompt_attributes(span)
        # Emit trace data to execution context queue if available
        self._emit_trace_to_execution_context(span, is_start=False)

        super().on_end(span)

    @staticmethod
    def _is_langfuse_span(span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name == LANGFUSE_TRACER_NAME
        )

    def _is_blocked_instrumentation_scope(self, span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name in self.blocked_instrumentation_scopes
        )

    def _is_langfuse_project_span(self, span: ReadableSpan) -> bool:
        if not LangfuseSpanProcessor._is_langfuse_span(span):
            return False

        if span.instrumentation_scope is not None:
            public_key_on_span = (
                span.instrumentation_scope.attributes.get("public_key", None)
                if span.instrumentation_scope.attributes
                else None
            )

            return public_key_on_span == self.public_key

        return False

    def _check_breakpoint(self, span: ReadableSpan | Span, is_before: bool) -> None:
        """Check and handle breakpoints with pause timing adjustment.

        This method ensures that pause duration is NOT counted in the span's duration
        by adjusting the span's start_time (for BEFORE pauses) or recording the pause
        as a separate event (for AFTER pauses).
        """
        # Get execution context
        breakpt_config = exec_ctx.get_current_breakpoint_config()

        # no breakpoint set
        if not breakpt_config:
            return

        # Check if timing matches
        if is_before != (breakpt_config.timing == "BEFORE"):
            return

        # Determine span type
        span_type = self._get_breakpoint_type(span)

        langfuse_logger.debug(
            "Breakpoint found at span: '%s', span type: %s, breakpoint config: %s",
            span.name,
            span_type,
            breakpt_config,
        )

        # Check if this span type should trigger a pause
        if span_type not in breakpt_config.breakpoint_types:
            return

        langfuse_logger.info(
            "Pausing execution at span '%s' (type=%s, timing=%s)",
            span.name,
            span_type,
            "BEFORE" if is_before else "AFTER",
        )

        # Mark span as pausible if it's a Span (not ReadableSpan)
        if isinstance(span, Span):
            span.set_attribute("pixie.pause.triggered", True)

        # Extract span attributes
        span_attributes = {}
        if hasattr(span, "attributes") and span.attributes:
            span_attributes = dict(span.attributes)

        breakpt = BreakpointDetail(
            span_name=span.name,
            breakpoint_type=span_type,
            breakpoint_timing="BEFORE" if is_before else "AFTER",
            span_attributes=span_attributes,
        )

        # Record pause start time (in nanoseconds to match OTel)
        pause_start_ns = time.time_ns()

        # Emit pause start event
        self._emit_pause_event(span, "pause_start", pause_start_ns)

        # Put paused event in queue (using asyncio)
        exec_ctx.emit_status_update(status="paused", breakpt=breakpt)

        # Wait for resume (this blocks)
        exec_ctx.wait_for_resume()

        # Record pause end time
        pause_end_ns = time.time_ns()
        pause_duration_ns = pause_end_ns - pause_start_ns

        langfuse_logger.info(
            "Resumed execution at span '%s', pause duration: %.3f seconds",
            span.name,
            pause_duration_ns / 1e9,
        )

        # Emit pause end event with duration
        self._emit_pause_event(
            span, "pause_end", pause_end_ns, pause_duration_ns=pause_duration_ns
        )

        # Adjust span timing to exclude pause duration
        if is_before and isinstance(span, Span):
            # For BEFORE pauses: adjust the span's start time forward
            # This ensures the pause time is NOT included in the span's duration
            # pylint: disable=protected-access
            if hasattr(span, "_start_time"):
                original_start = span._start_time
                span._start_time = original_start + pause_duration_ns
                # pylint: enable=protected-access
                langfuse_logger.debug(
                    "Adjusted span '%s' start_time by %d ns (%.3f seconds)",
                    span.name,
                    pause_duration_ns,
                    pause_duration_ns / 1e9,
                )
                # Add pause adjustment as span attribute
                span.set_attribute("pixie.pause.duration_ns", pause_duration_ns)
                span.set_attribute("pixie.pause.adjusted", True)
        elif not is_before and isinstance(span, ReadableSpan):
            # For AFTER pauses: the span has already ended, so just record the pause
            # The pause happens AFTER the span completes, so it doesn't affect duration
            # We emit the pause event separately
            langfuse_logger.debug(
                "Pause occurred AFTER span '%s' completed (duration not affected)",
                span.name,
            )

        exec_ctx.emit_status_update(status="running")

    def _get_breakpoint_type(
        self,
        span: ReadableSpan | Span,
    ) -> Optional[BreakpointType]:
        """Determine the pausible point type of a span."""
        # Check for custom pausible span attribute
        if hasattr(span, "attributes") and span.attributes:
            if span.attributes.get("pixie.pausible"):
                return "CUSTOM"

        # Check span name patterns for LLM calls
        span_name_lower = span.name.lower()
        if any(
            pattern in span_name_lower
            for pattern in ["openai", "llm", "chat", "completion", "generation"]
        ):
            return "LLM"

        # Check for tool/function calls
        if hasattr(span, "attributes") and span.attributes:
            # Check for tool execution attributes
            if (
                span.attributes.get("gen_ai.operation.name") == "tool"
                or "tool" in span_name_lower
                or "function_call" in span_name_lower
            ):
                return "TOOL"

        # Check span kind for client spans (often tool/API calls)
        if hasattr(span, "kind") and span.kind == SpanKind.CLIENT:
            if "tool" in span_name_lower or "function" in span_name_lower:
                return "TOOL"

        return None

    def _emit_trace_to_execution_context(
        self, span: ReadableSpan | Span, is_start: bool = False
    ) -> None:
        """Emit trace data to execution context queue for GraphQL subscription.

        This converts the span to OTLP JSON format (same as sent to Langfuse server)
        and emits it via the execution context queue if available.

        Args:
            span: The span to emit (ReadableSpan for completion, Span for start)
            is_start: True if emitting at span start, False if at span end
        """
        try:
            # Check if execution context exists using public API
            ctx = exec_ctx.get_current_context()
            if ctx is None:
                return

            if is_start:
                # For span start, emit partial trace data with available info
                trace_data = self._create_partial_trace_data(span)
            else:
                # For span end, convert full span to OTLP protobuf format
                proto_message = encode_spans([span])

                # Convert protobuf to JSON dict (same format as sent to Langfuse server)
                trace_data = MessageToDict(
                    proto_message,
                    preserving_proto_field_name=True,
                    always_print_fields_with_no_presence=True,
                )

            langfuse_logger.debug(
                "Emitting %s trace to execution context for span '%s'",
                "start" if is_start else "end",
                span.name,
            )

            # Emit using sync helper
            exec_ctx.emit_status_update(
                status="unchanged",
                trace=trace_data,
            )

        except Exception as e:  # pylint: disable=broad-except
            # Don't let trace emission failures break the span export
            langfuse_logger.warning(
                "Failed to emit trace to execution context: %s",
                str(e),
                exc_info=True,
            )

    def _create_partial_trace_data(self, span: ReadableSpan | Span) -> dict:
        """Create partial trace data for span start.

        Args:
            span: The span that is starting

        Returns:
            Dict with partial trace information available at span start
        """

        trace_data = {
            "event": "span_start",
            "span_name": span.name,
            "start_time_unix_nano": (
                str(span.start_time) if hasattr(span, "start_time") else None
            ),
        }

        # Add context information if available
        if hasattr(span, "context") and span.context:
            trace_data["trace_id"] = format_trace_id(span.context.trace_id)
            trace_data["span_id"] = format_span_id(span.context.span_id)

        # Add parent span id if available
        if hasattr(span, "parent") and span.parent:
            trace_data["parent_span_id"] = format_span_id(span.parent.span_id)

        # Add attributes if available
        if hasattr(span, "attributes") and span.attributes:
            trace_data["attributes"] = dict(span.attributes)

        # Add kind if available
        if hasattr(span, "kind"):
            trace_data["kind"] = span.kind.name

        return trace_data

    def _emit_pause_event(
        self,
        span: ReadableSpan | Span,
        event_type: str,
        event_time_ns: int,
        pause_duration_ns: Optional[int] = None,
    ) -> None:
        """Emit a pause-related event to both Langfuse and the subscription queue.

        Creates a custom trace event representing pause start/end and emits it via:
        1. The execution context queue (for GraphQL subscription updates)
        2. As span events in the OpenTelemetry span (exported to Langfuse)

        Args:
            span: The span where the pause occurred
            event_type: Type of pause event ("pause_start" or "pause_end")
            event_time_ns: Timestamp of the event in nanoseconds
            pause_duration_ns: Duration of the pause in nanoseconds (for pause_end events)
        """
        try:
            from opentelemetry.trace import format_trace_id

            # Create pause event data
            pause_event = {
                "event": event_type,
                "timestamp_ns": event_time_ns,
                "span_name": span.name,
            }

            # Add context information if available
            if hasattr(span, "context") and span.context:
                pause_event["trace_id"] = format_trace_id(span.context.trace_id)
                pause_event["span_id"] = format_span_id(span.context.span_id)

            # Add parent span id if available
            if hasattr(span, "parent") and span.parent:
                pause_event["parent_span_id"] = format_span_id(span.parent.span_id)

            # Add pause duration for end events
            if pause_duration_ns is not None:
                pause_event["pause_duration_ns"] = pause_duration_ns
                pause_event["pause_duration_seconds"] = pause_duration_ns / 1e9

            # Add span attributes if available
            if hasattr(span, "attributes") and span.attributes:
                pause_event["span_attributes"] = dict(span.attributes)

            langfuse_logger.debug(
                "Emitting pause event '%s' for span '%s'",
                event_type,
                span.name,
            )

            # 1. Emit to execution context queue for subscription updates
            ctx = exec_ctx.get_current_context()
            if ctx is not None:
                exec_ctx.emit_status_update(
                    status="unchanged",
                    trace=pause_event,
                )

            # 2. Add as OpenTelemetry span event (will be exported to Langfuse)
            if isinstance(span, Span):
                # Add event attributes
                event_attributes = {
                    "pixie.event.type": event_type,
                    "pixie.pause.timestamp_ns": event_time_ns,
                }
                if pause_duration_ns is not None:
                    event_attributes["pixie.pause.duration_ns"] = pause_duration_ns
                    event_attributes["pixie.pause.duration_seconds"] = (
                        pause_duration_ns / 1e9
                    )

                # Add event to span
                span.add_event(
                    name=f"pixie.{event_type}",
                    attributes=event_attributes,
                    timestamp=event_time_ns,
                )

                langfuse_logger.debug(
                    "Added '%s' event to span '%s' (will be exported to Langfuse)",
                    event_type,
                    span.name,
                )

        except Exception as e:  # pylint: disable=broad-except
            # Don't let pause event emission failures break the pause/resume flow
            langfuse_logger.warning(
                "Failed to emit pause event: %s",
                str(e),
                exc_info=True,
            )

    def _emit_prompt_attributes(self, span: ReadableSpan) -> None:
        if not span.attributes or not span.context:
            return
        for attr_value in span.attributes.values():
            if not isinstance(attr_value, str):
                continue
            associated_prompt = get_compiled_prompt(attr_value)
            if associated_prompt:
                langfuse_logger.debug(
                    "Emitting Pixie prompt attributes for span '%s' prompt id '%s'",
                    span.name,
                    associated_prompt.prompt.id,
                )
                update = PromptForSpan(
                    trace_id=format_trace_id(span.context.trace_id),
                    span_id=format_span_id(span.context.span_id),
                    prompt_id=associated_prompt.prompt.id,
                    version_id=associated_prompt.version_id,
                    variables=(
                        associated_prompt.variables.model_dump(mode="json")
                        if associated_prompt.variables
                        else None
                    ),
                )
                exec_ctx.emit_status_update(status="unchanged", prompt_for_span=update)
