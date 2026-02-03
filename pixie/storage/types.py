"""Storage types for run and LLM call records.

These types are used for:
1. Database storage (via Piccolo ORM)
2. GraphQL API (via Strawberry conversion)
3. Data exchange between frontend and backend
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, JsonValue

# Rating type used across the system
Rating = Literal["good", "bad", "undecided"]
RunSource = Literal["apps", "sessions"]


class Message(BaseModel):
    """A message in interaction logs."""

    role: Literal["system", "user", "assistant", "tool"]
    content: JsonValue
    time_unix_nano: str | None = None
    user_rating: Rating | None = None
    user_feedback: str | None = None


class RatingDetails(BaseModel):
    """Rating details for a run or LLM call."""

    value: Rating
    rated_at: str  # Unix timestamp in milliseconds as string
    notes: str | None = None


class AppInfoRecord(BaseModel):
    """App information stored with records."""

    id: str
    name: str
    module: str
    qualified_name: str
    short_description: str | None = None
    full_description: str | None = None


class SessionInfoRecord(BaseModel):
    """Session information stored with records."""

    session_id: str
    name: str
    module: str
    qualname: str
    description: str | None = None


class PromptInfoRecord(BaseModel):
    """Prompt information for an LLM call."""

    prompt_id: str
    version_id: str
    variables: dict[str, JsonValue] | None = None


class RunRecordData(BaseModel):
    """Core data for a run record.

    This model is used for both creating and updating run records.
    All fields except id are optional for partial updates.
    """

    id: str = Field(description="Run ID for apps, session ID for sessions")
    source: RunSource = Field(
        default="apps", description="Source type: 'apps' or 'sessions'"
    )
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    messages: list[Message] = Field(default_factory=list)
    prompt_ids: list[str] = Field(default_factory=list)
    start_time: str | None = None  # Unix timestamp in milliseconds as string
    end_time: str | None = None  # Unix timestamp in milliseconds as string
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RunRecordInput(BaseModel):
    """Input for creating a run record."""

    id: str
    source: RunSource
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    messages: list[Message] = Field(default_factory=list)
    prompt_ids: list[str] = Field(default_factory=list)
    start_time: str | None = None  # Unix timestamp in milliseconds as string
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RunRecordUpdate(BaseModel):
    """Input for updating a run record (partial update)."""

    messages: list[Message] | None = None
    prompt_ids: list[str] | None = None
    end_time: str | None = None  # Unix timestamp in milliseconds as string
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue] | None = None


class RunRecord(BaseModel):
    """Full run record with timestamps."""

    id: str
    source: RunSource
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    messages: list[Message]
    prompt_ids: list[str]
    start_time: str | None = None
    end_time: str | None = None
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue]
    created_at: datetime
    updated_at: datetime


class LlmCallRecordData(BaseModel):
    """Core data for an LLM call record."""

    span_id: str = Field(description="Unique span ID for this LLM call")
    trace_id: str
    run_id: str | None = None  # Link to parent run record
    run_source: RunSource = "apps"
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    prompt_info: PromptInfoRecord | None = None
    llm_input: JsonValue = None
    llm_output: JsonValue = None
    model_name: str = "unknown"
    model_parameters: dict[str, JsonValue] | None = None
    start_time: str | None = None  # Unix timestamp in nanoseconds as string
    end_time: str | None = None  # Unix timestamp in nanoseconds as string
    internal_logs_after: list[JsonValue] = Field(default_factory=list)
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class LlmCallRecordInput(BaseModel):
    """Input for creating an LLM call record."""

    span_id: str
    trace_id: str
    run_id: str | None = None
    run_source: RunSource = "apps"
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    prompt_info: PromptInfoRecord | None = None
    llm_input: JsonValue = None
    llm_output: JsonValue = None
    model_name: str = "unknown"
    model_parameters: dict[str, JsonValue] | None = None
    start_time: str | None = None  # Unix timestamp in nanoseconds as string
    end_time: str | None = None  # Unix timestamp in nanoseconds as string
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class LlmCallRecordUpdate(BaseModel):
    """Input for updating an LLM call record (partial update)."""

    prompt_info: PromptInfoRecord | None = None
    llm_input: JsonValue | None = None
    llm_output: JsonValue | None = None
    model_name: str | None = None
    model_parameters: dict[str, JsonValue] | None = None
    end_time: str | None = None  # Unix timestamp in nanoseconds as string
    internal_logs_after: list[JsonValue] | None = None
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue] | None = None


class LlmCallRecord(BaseModel):
    """Full LLM call record with timestamps."""

    span_id: str
    trace_id: str
    run_id: str | None = None
    run_source: RunSource
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    prompt_info: PromptInfoRecord | None = None
    llm_input: JsonValue
    llm_output: JsonValue
    model_name: str
    model_parameters: dict[str, JsonValue] | None = None
    start_time: str | None = None
    end_time: str | None = None
    internal_logs_after: list[JsonValue]
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue]
    created_at: datetime
    updated_at: datetime


class RecordFilters(BaseModel):
    """Common filters for querying records."""

    prompt_id: str | None = None
    app_id: str | None = None
    run_source: RunSource | None = None
    has_rating: bool | None = None
    rating_value: Rating | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 100
    offset: int = 0
