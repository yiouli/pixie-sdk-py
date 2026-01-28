"""Type definitions for SDK."""

import threading
from dataclasses import dataclass
from typing import AsyncGenerator, Generic, Literal, Optional, Any, TypeVar
import janus
from pydantic import BaseModel, JsonValue


# Forward import to avoid circular dependency
# The actual types are defined in otel_types.py
TraceDataType = dict[str, Any]  # Will be OTLPTraceData | PartialTraceData


class ExecutionInput(BaseModel):
    """Input for agent execution.

    Attributes:
        input_data: JSON string containing the input data for the agent.
        context: Optional JSON string containing additional context.
    """

    input_data: str  # JSON string
    context: Optional[str] = None  # JSON string


class ExecutionStatus(BaseModel):
    """Status update for agent execution.

    Attributes:
        status: Current status of the execution (running, completed, or error).
        message: Human-readable message describing the status.
        data: Optional JSON string containing execution data.
        error: Optional error message if status is error.
    """

    status: Literal["running", "completed", "error"]
    message: str
    data: Optional[str] = None  # JSON string
    error: Optional[str] = None


# Pause/Resume Types
BreakpointTiming = Literal["BEFORE", "AFTER"]
BreakpointType = Literal["LLM", "TOOL", "CUSTOM"]


class BreakpointConfig(BaseModel):
    """Configuration for pause behavior.

    Attributes:
        id: Unique identifier for this breakpoint configuration.
        timing: When to pause (BEFORE or AFTER the breakpoint).
        breakpoint_types: List of breakpoint types to pause on (LLM, TOOL, CUSTOM).
    """

    id: str
    timing: BreakpointTiming
    breakpoint_types: list[BreakpointType]


class PauseResult(BaseModel):
    """Result of a pause operation.

    Attributes:
        success: Whether the pause operation was successful.
        message: Human-readable message about the pause operation.
        run_id: Optional identifier of the paused run.
        paused_state: Optional dictionary containing the paused execution state.
    """

    success: bool
    message: str
    run_id: Optional[str] = None
    paused_state: Optional[dict] = None


class ResumeResult(BaseModel):
    """Result of a resume operation.

    Attributes:
        success: Whether the resume operation was successful.
        message: Human-readable message about the resume operation.
        run_id: Optional identifier of the resumed run.
    """

    success: bool
    message: str
    run_id: Optional[str] = None


class BreakpointDetail(BaseModel):
    """Information about a breakpoint where execution paused.

    Attributes:
        span_name: Name of the span where execution paused.
        breakpoint_type: Type of breakpoint (LLM, TOOL, or CUSTOM).
        breakpoint_timing: When the pause occurred (BEFORE or AFTER).
        span_attributes: Optional dictionary of attributes from the span.
    """

    span_name: str
    breakpoint_type: BreakpointType
    breakpoint_timing: BreakpointTiming
    span_attributes: Optional[dict] = None


RunStatus = Literal[
    "running",
    "paused",
    "completed",
    "error",
    "cancelled",
    "waiting",
]

# TypeVars for decorator overloads
InputType = TypeVar(
    "InputType", bound=BaseModel | JsonValue
)  # Input parameter type (covariant-like)
_OutputType = TypeVar(
    "_OutputType", bound=BaseModel | JsonValue
)  # Return type (covariant)


class InputRequired(Generic[InputType]):
    """Represents a requirement for user input during application execution.

    This is yielded by a generator application to request input from the user.
    Can be initialized with either a type hint or a JSON schema dict directly.

    Attributes:
        expected_type: The type of input expected from the user (None if schema provided).
        json_schema: The JSON schema for the expected input (None if type provided).
    """

    def __init__(
        self,
        expected_type_or_schema: type[InputType] | dict[str, Any],
    ) -> None:
        """Initialize a user input requirement.

        Args:
            expected_type_or_schema: Either a type hint (e.g., str, int, BaseModel)
                or a JSON schema dict directly.
        """
        if isinstance(expected_type_or_schema, dict):
            # Direct JSON schema provided
            self.expected_type: type[InputType] | None = None
            self.json_schema: dict[str, Any] | None = expected_type_or_schema
        else:
            # Type hint provided
            self.expected_type = expected_type_or_schema
            self.json_schema = None

    def get_json_schema(self) -> dict[str, Any] | None:
        """Get the JSON schema for this input requirement.

        Returns:
            The JSON schema dict, either directly provided or derived from the type.
        """
        if self.json_schema is not None:
            return self.json_schema

        if self.expected_type is None:
            return None

        # Import here to avoid circular dependency
        from pixie.utils import get_json_schema_for_type

        return get_json_schema_for_type(self.expected_type)


class PromptForSpan(BaseModel):
    """Information about the prompt used in the application run.

    Attributes:
        prompt_id: Unique identifier of the prompt.
        version_id: Version identifier of the prompt.
        variables: Optional variables used in the prompt.
    """

    trace_id: str
    span_id: str
    prompt_id: str
    version_id: str
    variables: Optional[dict[str, JsonValue]] = None


class InputMixin(BaseModel):
    """Mixin providing user input handling fields.

    Attributes:
        user_input: Optional user input data received.
        user_input_schema: Optional JSON schema describing expected input format.
    """

    user_input: Optional[JsonValue] = None
    user_input_schema: Optional[JsonValue] = None


class OutputMixin(BaseModel):
    data: Optional[JsonValue] = None
    trace: Optional[TraceDataType] = None
    prompt_for_span: Optional[PromptForSpan] = None


class AppRunUpdate(OutputMixin, InputMixin):
    """Status update from running an application.

    Attributes:
        run_id: Unique identifier of the application run.
        status: Current status of the run.
        user_input: Optional user input that was received.
        data: Optional output data from the application.
        breakpoint: Optional[BreakpointDetail] = None
        trace: Optional[TraceDataType] = None
        prompt_for_span: Optional[PromptForSpan] = None
    """

    run_id: str
    status: RunStatus
    breakpoint: Optional[BreakpointDetail] = None


@dataclass
class ExecutionContext:
    """Context for a running execution.

    Attributes:
        run_id: Unique identifier of the execution run.
        status_queue: Queue for passing status updates (None signals end of stream).
        resume_event: Threading event for pause/resume functionality.
        breakpoint_config: Optional configuration for execution breakpoints.
        cancelled: Flag indicating if the execution has been cancelled.
    """

    run_id: str
    # None is the sentinel to end the stream
    status_queue: janus.Queue[AppRunUpdate | None]
    resume_event: threading.Event
    breakpoint_config: Optional[BreakpointConfig] = None
    cancelled: bool = False


class AppRunCancelled(Exception):
    """Exception raised when an application run is cancelled.

    This exception is thrown when an application execution is cancelled
    either explicitly by the user or due to a cancellation request.
    """


PixieGenerator = AsyncGenerator[str | _OutputType | InputRequired[InputType], InputType]
"""Async generator that yields streaming token, output or input requirements and receives user input."""
