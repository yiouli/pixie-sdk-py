"""Shared Strawberry types and enums for GraphQL schema.

This module contains all shared enums and types that are used across
multiple parts of the GraphQL schema to avoid circular imports.
"""

from enum import Enum
import strawberry


@strawberry.enum
class BreakpointTiming(str, Enum):
    """Mode for pausing execution."""

    BEFORE = "BEFORE"
    AFTER = "AFTER"


@strawberry.enum
class BreakpointType(str, Enum):
    """Types of pausible points in execution."""

    LLM = "LLM"
    TOOL = "TOOL"
    CUSTOM = "CUSTOM"


@strawberry.enum
class AppRunStatus(str, Enum):
    """Status of an application run."""

    running = "running"
    completed = "completed"
    error = "error"
    paused = "paused"
    waiting = "waiting"
    cancelled = "cancelled"
    unchanged = "unchanged"


@strawberry.enum
class Rating(str, Enum):
    """Rating for an LLM call or app run."""

    good = "good"
    bad = "bad"
    undecided = "undecided"


@strawberry.enum
class MessageRole(str, Enum):
    """Role of a message in interaction logs."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    developer = "developer"


@strawberry.enum
class RunSource(str, Enum):
    """Source type for run records."""

    apps = "apps"
    sessions = "sessions"


@strawberry.enum
class TraceEventType(str, Enum):
    """Type of trace event.

    Indicates whether the trace data is from a span starting or other event.
    """

    SPAN_START = "span_start"
