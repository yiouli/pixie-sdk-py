"""Type definitions for SDK."""

import asyncio
import threading
from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import BaseModel


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
PauseModeType = Literal["BEFORE", "AFTER"]
PausiblePointType = Literal["LLM", "TOOL", "CUSTOM"]


class PauseConfig(BaseModel):
    """Configuration for pause behavior."""

    mode: PauseModeType
    pausible_points: list[PausiblePointType]


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


class Breakpoint(BaseModel):
    """Information about a breakpoint where execution paused."""

    span_name: str
    breakpoint_type: PausiblePointType
    breakpoint_timing: PauseModeType
    span_attributes: Optional[dict] = None


class AppRunUpdate(BaseModel):
    """Status update from running an application."""

    run_id: str
    status: str
    data: Optional[str] = None
    breakpoint: Optional[Breakpoint] = None


@dataclass
class ExecutionContext:
    """Context for a running execution."""

    run_id: str
    pause_config: Optional[PauseConfig] = None
    status_queue: Optional["asyncio.Queue[AppRunUpdate]"] = None
    resume_event: Optional[asyncio.Event] = None
    pause_start_time: Optional[float] = None
    # Threading event for synchronous waiting
    sync_resume_event: Optional["threading.Event"] = None
