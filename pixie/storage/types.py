"""Storage types for run and LLM call records.

These types are used for:
1. Database storage (via Piccolo ORM)
2. GraphQL API (via Strawberry conversion)
3. Data exchange between frontend and backend

Type hierarchy:
- RunRecordDetails: Base class with all updatable fields (all optional)
- RunRecord: Full record with required id (extends RunRecordDetails)

- LlmCallRecordDetails: Base class with all updatable fields (all optional)
- LlmCallRecord: Full record with required span_id, trace_id (extends LlmCallRecordDetails)
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, JsonValue

# Type aliases for string literal values
Rating = Literal["good", "bad", "undecided"]
RatedBy = Literal["user", "ai", "system"]
RunSource = Literal["apps", "sessions"]
MessageRole = Literal["system", "user", "assistant", "tool", "developer"]


class Message(BaseModel):
    """A message in interaction logs."""

    role: MessageRole
    content: JsonValue
    time_unix_nano: str | None = None
    user_rating: Rating | None = None
    user_feedback: str | None = None


class RatingDetails(BaseModel):
    """Rating details for a run or LLM call."""

    value: Rating
    rated_at: str  # Unix timestamp in milliseconds as string
    rated_by: RatedBy
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


# ============================================================================
# Run Record Types
# ============================================================================


class RunRecordDetails(BaseModel):
    """Details for a run record (used for updates).

    All fields are optional to support partial updates.
    """

    source: RunSource | None = None
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    messages: list[Message] | None = None
    prompt_ids: list[str] | None = None
    start_time: str | None = None  # Unix timestamp in milliseconds as string
    end_time: str | None = None  # Unix timestamp in milliseconds as string
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue] | None = None


class RunRecord(RunRecordDetails):
    """Full run record with required fields and timestamps.

    Used for both creating records (input) and returning records (output).
    The id field is required.
    """

    id: str = Field(description="Run ID for apps, session ID for sessions")
    # DB timestamps (set by database, not user)
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ============================================================================
# LLM Call Record Types
# ============================================================================


class ToolDefinition(BaseModel):
    """Tool/function definition for LLM function calling."""

    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None


class LlmCallRecordDetails(BaseModel):
    """Details for an LLM call record (used for updates).

    All fields are optional to support partial updates.
    """

    run_id: str | None = None  # Link to parent run record
    run_source: RunSource | None = None  # Optional since run_id might be None
    app_info: AppInfoRecord | None = None
    session_info: SessionInfoRecord | None = None
    prompt_info: PromptInfoRecord | None = None
    llm_input: JsonValue = None
    llm_output: JsonValue = None
    tools: list[ToolDefinition] | None = None  # Tool definitions for function calling
    output_type: dict[str, JsonValue] | None = None  # JSON schema for structured output
    model_name: str | None = None
    model_parameters: dict[str, JsonValue] | None = None
    start_time: str | None = None  # Unix timestamp in nanoseconds as string
    end_time: str | None = None  # Unix timestamp in nanoseconds as string
    internal_logs_after: list[JsonValue] | None = None
    rating: RatingDetails | None = None
    metadata: dict[str, JsonValue] | None = None


class LlmCallRecord(LlmCallRecordDetails):
    """Full LLM call record with required fields.

    Used for both creating records (input) and returning records (output).
    """

    span_id: str = Field(description="Unique span ID for this LLM call")
    trace_id: str


# ============================================================================
# Query Filters
# ============================================================================


class RecordFilters(BaseModel):
    """Common filters for querying records."""

    prompt_id: str | None = None
    app_ids: list[str] | None = None
    run_source: RunSource | None = None
    has_rating: bool | None = None
    rating_values: list[Rating] | None = None
    rated_by_values: list[RatedBy] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 100
    offset: int = 0


# Type aliases for backward compatibility with tests
# The "input" type is the same as the full record type (with required id/span_id)
# The "update" type is the details type (all fields optional)
RunRecordInput = RunRecord
RunRecordUpdate = RunRecordDetails
LlmCallRecordInput = LlmCallRecord
LlmCallRecordUpdate = LlmCallRecordDetails
