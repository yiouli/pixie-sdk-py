"""Type definitions for SDK."""

import threading
from dataclasses import dataclass
from typing import AsyncGenerator, Generic, Literal, Optional, Any, TypeVar
import janus
from pydantic import BaseModel, JsonValue, PrivateAttr


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


AppRunStatus = Literal[
    "running",
    "paused",
    "completed",
    "error",
    "cancelled",
    "waiting",
]

# TypeVars for decorator overloads
_UserInputType = TypeVar(
    "_UserInputType", bound=BaseModel | JsonValue
)  # Input parameter type (covariant-like)
_OutputType = TypeVar(
    "_OutputType", bound=BaseModel | JsonValue
)  # Return type (covariant)


class InputRequired(Generic[_UserInputType]):
    """Represents a requirement for user input during application execution.

    This is yielded by a generator application to request input from the user.

    Attributes:
        expected_type: The type of input expected from the user.
    """

    def __init__(self, expected_type: type[_UserInputType]) -> None:
        """Initialize a user input requirement.

        Args:
            expected_type: The type of input expected from the user.
        """
        self.expected_type = expected_type


class AppRunUpdate(BaseModel):
    """Status update from running an application.

    Attributes:
        run_id: Unique identifier of the application run.
        status: Current status of the run.
        user_input: Optional user input that was received.
        data: Optional output data from the application.
        breakpoint: Optional details about a breakpoint if execution paused.
        trace: Optional trace data for observability.
    """

    run_id: str
    status: AppRunStatus
    user_input: Optional[JsonValue] = None
    data: Optional[JsonValue] = None
    breakpoint: Optional[BreakpointDetail] = None
    trace: Optional[TraceDataType] = None
    _user_input_requirement: InputRequired | None = PrivateAttr(default=None)

    def set_user_input_requirement(
        self,
        requirement: InputRequired | None,
    ) -> None:
        """Set the user input requirement for this update.

        Args:
            requirement: The user input requirement or None.
        """
        self._user_input_requirement = requirement

    @property
    def user_input_requirement(self) -> InputRequired | None:
        """Get the user input requirement for this update.

        Returns:
            The user input requirement or None.
        """
        return self._user_input_requirement


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


PixieGenerator = AsyncGenerator[
    str | _OutputType | InputRequired[_UserInputType], _UserInputType
]
"""Async generator that yields streaming token, output or input requirements and receives user input."""
