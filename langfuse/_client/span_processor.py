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

import asyncio
import base64
import os
from typing import Dict, List, Optional

from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, SpanKind

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
from pixie.types import BreakpointDetail, BreakpointType


class LangfuseSpanProcessor(BatchSpanProcessor):
    """OpenTelemetry span processor that exports spans to the Langfuse API.

    This processor extends OpenTelemetry's BatchSpanProcessor with Langfuse-specific functionality:
    1. Project-scoped span filtering to prevent cross-project data leakage
    2. Instrumentation scope filtering to block spans from specific libraries/frameworks
    3. Configurable batch processing parameters for optimal performance
    4. HTTP-based span export to the Langfuse OTLP endpoint
    5. Debug logging for span processing operations
    6. Authentication with Langfuse API using Basic Auth

    The processor is designed to efficiently handle large volumes of spans with
    minimal overhead, while ensuring spans are only sent to the correct project.
    It integrates with OpenTelemetry's standard span lifecycle, adding Langfuse-specific
    filtering and export capabilities.
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
    ):
        self.public_key = public_key
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
        try:
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

            # Put paused event in queue (using asyncio)
            asyncio.run(exec_ctx.emit_status_update(status="paused", breakpt=breakpt))
            exec_ctx.wait_for_resume()
            asyncio.run(exec_ctx.emit_status_update(status="running"))

        except Exception as e:
            # Don't break application due to pause infrastructure errors
            langfuse_logger.error(
                "Pause logic failed for span '%s': %s", span.name, str(e)
            )

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
