"""Type definitions for SDK."""

import asyncio
import threading
from dataclasses import dataclass
from typing import AsyncGenerator, Generic, Literal, Optional, Any, TypeVar
from pydantic import BaseModel, JsonValue, SkipValidation


# Forward import to avoid circular dependency
# The actual types are defined in otel_types.py
TraceDataType = dict[str, Any]  # Will be OTLPTraceData | PartialTraceData


class ExecutionInput(BaseModel):
    """Input for agent execution."""

    input_data: str  # JSON string
    context: Optional[str] = None  # JSON string


class ExecutionStatus(BaseModel):
    """Status update for agent execution."""

    status: Literal["running", "completed", "error"]
    message: str
    data: Optional[str] = None  # JSON string
    error: Optional[str] = None


# Pause/Resume Types
BreakpointTiming = Literal["BEFORE", "AFTER"]
BreakpointType = Literal["LLM", "TOOL", "CUSTOM"]


class BreakpointConfig(BaseModel):
    """Configuration for pause behavior."""

    id: str
    timing: BreakpointTiming
    breakpoint_types: list[BreakpointType]


class PauseResult(BaseModel):
    """Result of a pause operation."""

    success: bool
    message: str
    run_id: Optional[str] = None
    paused_state: Optional[dict] = None


class ResumeResult(BaseModel):
    """Result of a resume operation."""

    success: bool
    message: str
    run_id: Optional[str] = None


class BreakpointDetail(BaseModel):
    """Information about a breakpoint where execution paused."""

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


class UserInputRequirement(Generic[_UserInputType]):
    def __init__(self, expected_type: type[_UserInputType]) -> None:
        self.expected_type = expected_type


class AppRunUpdate(BaseModel):
    """Status update from running an application."""

    run_id: str
    status: AppRunStatus
    user_input_requirement: SkipValidation[UserInputRequirement | None] = None
    user_input: Optional[JsonValue] = None
    data: Optional[JsonValue] = None
    breakpoint: Optional[BreakpointDetail] = None
    trace: Optional[TraceDataType] = None


@dataclass
class ExecutionContext:
    """Context for a running execution."""

    run_id: str
    # None is the sentinel to end the stream
    status_queue: asyncio.Queue[AppRunUpdate | None]
    resume_event: threading.Event
    breakpoint_config: Optional[BreakpointConfig] = None
    cancelled: bool = False


class AppRunCancelled(Exception):
    """Exception raised when an application run is cancelled."""


PixieGenerator = AsyncGenerator[
    UserInputRequirement[_UserInputType] | str | _OutputType, _UserInputType
]
