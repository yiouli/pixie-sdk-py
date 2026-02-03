"""GraphQL types and resolvers for storage records."""

from datetime import datetime
from enum import Enum
from typing import Optional

import strawberry
from strawberry.scalars import JSON
import strawberry.experimental.pydantic

from pixie.storage.types import (
    Message as PydanticMessage,
    RatingDetails as PydanticRatingDetails,
    AppInfoRecord as PydanticAppInfoRecord,
    SessionInfoRecord as PydanticSessionInfoRecord,
    PromptInfoRecord as PydanticPromptInfoRecord,
    RunRecord as PydanticRunRecord,
    RunRecordInput as PydanticRunRecordInput,
    RunRecordUpdate as PydanticRunRecordUpdate,
    LlmCallRecord as PydanticLlmCallRecord,
    LlmCallRecordInput as PydanticLlmCallRecordInput,
    LlmCallRecordUpdate as PydanticLlmCallRecordUpdate,
    RecordFilters as PydanticRecordFilters,
)
from pixie.storage import (
    create_run_record as db_create_run_record,
    get_run_record as db_get_run_record,
    update_run_record as db_update_run_record,
    get_run_records as db_get_run_records,
    create_llm_call_record as db_create_llm_call_record,
    get_llm_call_record as db_get_llm_call_record,
    update_llm_call_record as db_update_llm_call_record,
    get_llm_call_records as db_get_llm_call_records,
    get_llm_call_records_by_run as db_get_llm_call_records_by_run,
)

# ============================================================================
# Enums
# ============================================================================


class RunSourceEnum(Enum):
    """Source type for run records."""

    APPS = "apps"
    SESSIONS = "sessions"


class RatingValueEnum(Enum):
    """Rating value."""

    GOOD = "good"
    BAD = "bad"
    UNDECIDED = "undecided"


# ============================================================================
# Output Types (Pydantic to Strawberry conversion)
# ============================================================================


@strawberry.experimental.pydantic.type(model=PydanticRatingDetails)
class RatingDetailsType:
    """Rating details for a record."""

    value: str  # Literal type not supported, use str
    rated_at: str  # String to avoid 32-bit int overflow
    notes: strawberry.auto


@strawberry.experimental.pydantic.type(model=PydanticAppInfoRecord, all_fields=True)
class AppInfoRecordType:
    """App information in a record."""

    pass


@strawberry.experimental.pydantic.type(model=PydanticSessionInfoRecord, all_fields=True)
class SessionInfoRecordType:
    """Session information in a record."""

    pass


@strawberry.experimental.pydantic.type(model=PydanticPromptInfoRecord)
class PromptInfoRecordType:
    """Prompt information in a record."""

    prompt_id: strawberry.auto
    version_id: strawberry.auto
    variables: Optional[JSON] = None


@strawberry.experimental.pydantic.type(model=PydanticMessage)
class StorageMessage:
    """Message in interaction logs (for storage records)."""

    role: str  # Literal type not supported, use str
    content: JSON
    time_unix_nano: Optional[str] = None
    user_rating: Optional[str] = None  # Literal type not supported
    user_feedback: Optional[str] = None


@strawberry.experimental.pydantic.type(model=PydanticRunRecord)
class RunRecordType:
    """Full run record."""

    id: strawberry.auto
    source: str  # Literal type not supported, use str
    app_info: Optional[AppInfoRecordType] = None
    session_info: Optional[SessionInfoRecordType] = None
    messages: list[StorageMessage]
    prompt_ids: list[str]
    start_time: Optional[str] = None  # String to avoid 32-bit int overflow
    end_time: Optional[str] = None  # String to avoid 32-bit int overflow
    rating: Optional[RatingDetailsType] = None
    metadata: JSON
    created_at: datetime
    updated_at: datetime


@strawberry.experimental.pydantic.type(model=PydanticLlmCallRecord)
class LlmCallRecordType:
    """Full LLM call record."""

    span_id: strawberry.auto
    trace_id: strawberry.auto
    run_id: Optional[str] = None
    run_source: str  # Literal type not supported, use str
    app_info: Optional[AppInfoRecordType] = None
    session_info: Optional[SessionInfoRecordType] = None
    prompt_info: Optional[PromptInfoRecordType] = None
    llm_input: JSON
    llm_output: JSON
    model_name: strawberry.auto
    model_parameters: Optional[JSON] = None
    start_time: Optional[str] = None  # String to avoid 32-bit int overflow
    end_time: Optional[str] = None  # String to avoid 32-bit int overflow
    internal_logs_after: list[JSON]
    rating: Optional[RatingDetailsType] = None
    metadata: JSON
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Input Types
# ============================================================================


@strawberry.experimental.pydantic.input(model=PydanticRatingDetails)
class RatingDetailsInput:
    """Rating details input."""

    value: str  # Literal type not supported, use str
    rated_at: str  # String to avoid 32-bit int overflow
    notes: strawberry.auto


@strawberry.experimental.pydantic.input(model=PydanticAppInfoRecord, all_fields=True)
class AppInfoRecordInput:
    """App information input."""

    pass


@strawberry.experimental.pydantic.input(
    model=PydanticSessionInfoRecord, all_fields=True
)
class SessionInfoRecordInput:
    """Session information input."""

    pass


@strawberry.experimental.pydantic.input(model=PydanticPromptInfoRecord)
class PromptInfoRecordInput:
    """Prompt information input."""

    prompt_id: strawberry.auto
    version_id: strawberry.auto
    variables: Optional[JSON] = None


@strawberry.experimental.pydantic.input(model=PydanticMessage)
class StorageMessageInput:
    """Message input for storage records."""

    role: str  # Literal type not supported, use str
    content: JSON
    time_unix_nano: Optional[str] = None
    user_rating: Optional[str] = None  # Literal type not supported
    user_feedback: Optional[str] = None


@strawberry.experimental.pydantic.input(model=PydanticRunRecordInput)
class RunRecordInputType:
    """Input for creating a run record."""

    id: strawberry.auto
    source: str  # Literal type not supported, use str
    app_info: Optional[AppInfoRecordInput] = None
    session_info: Optional[SessionInfoRecordInput] = None
    messages: list[StorageMessageInput] = strawberry.field(default_factory=list)
    prompt_ids: list[str] = strawberry.field(default_factory=list)
    start_time: Optional[str] = None  # String to avoid 32-bit int overflow
    metadata: JSON = strawberry.field(default_factory=dict)


@strawberry.experimental.pydantic.input(model=PydanticRunRecordUpdate)
class RunRecordUpdateInput:
    """Input for updating a run record."""

    messages: Optional[list[StorageMessageInput]] = None
    prompt_ids: Optional[list[str]] = None
    end_time: Optional[str] = None  # String to avoid 32-bit int overflow
    rating: Optional[RatingDetailsInput] = None
    metadata: Optional[JSON] = None


@strawberry.experimental.pydantic.input(model=PydanticLlmCallRecordInput)
class LlmCallRecordInputType:
    """Input for creating an LLM call record."""

    span_id: strawberry.auto
    trace_id: strawberry.auto
    run_id: Optional[str] = None
    run_source: str = "apps"  # Literal type not supported, use str
    app_info: Optional[AppInfoRecordInput] = None
    session_info: Optional[SessionInfoRecordInput] = None
    prompt_info: Optional[PromptInfoRecordInput] = None
    llm_input: Optional[JSON] = None
    llm_output: Optional[JSON] = None
    model_name: str = "unknown"
    model_parameters: Optional[JSON] = None
    start_time: Optional[str] = None  # String to avoid 32-bit int overflow
    end_time: Optional[str] = None  # String to avoid 32-bit int overflow
    metadata: JSON = strawberry.field(default_factory=dict)


@strawberry.experimental.pydantic.input(model=PydanticLlmCallRecordUpdate)
class LlmCallRecordUpdateInput:
    """Input for updating an LLM call record."""

    prompt_info: Optional[PromptInfoRecordInput] = None
    llm_input: Optional[JSON] = None
    llm_output: Optional[JSON] = None
    model_name: Optional[str] = None
    model_parameters: Optional[JSON] = None
    end_time: Optional[str] = None  # String to avoid 32-bit int overflow
    internal_logs_after: Optional[list[JSON]] = None
    rating: Optional[RatingDetailsInput] = None
    metadata: Optional[JSON] = None


@strawberry.experimental.pydantic.input(model=PydanticRecordFilters)
class RecordFiltersInput:
    """Filters for querying records."""

    prompt_id: Optional[str] = None
    app_id: Optional[str] = None
    run_source: Optional[str] = None
    has_rating: Optional[bool] = None
    rating_value: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = 100
    offset: int = 0


# ============================================================================
# GraphQL Query
# ============================================================================


@strawberry.type
class StorageQuery:
    """GraphQL queries for storage records."""

    @strawberry.field
    async def get_run_record(self, run_id: str) -> Optional[RunRecordType]:
        """Get a run record by ID."""
        record = await db_get_run_record(run_id)
        if record is None:
            return None
        return RunRecordType.from_pydantic(record)

    @strawberry.field
    async def get_run_records(
        self, filters: Optional[RecordFiltersInput] = None
    ) -> list[RunRecordType]:
        """Get run records with optional filters."""
        filter_model = filters.to_pydantic() if filters else PydanticRecordFilters()
        records = await db_get_run_records(filter_model)
        return [RunRecordType.from_pydantic(r) for r in records]

    @strawberry.field
    async def get_llm_call_record(self, span_id: str) -> Optional[LlmCallRecordType]:
        """Get an LLM call record by span ID."""
        record = await db_get_llm_call_record(span_id)
        if record is None:
            return None
        return LlmCallRecordType.from_pydantic(record)

    @strawberry.field
    async def get_llm_call_records(
        self, filters: Optional[RecordFiltersInput] = None
    ) -> list[LlmCallRecordType]:
        """Get LLM call records with optional filters."""
        filter_model = filters.to_pydantic() if filters else PydanticRecordFilters()
        records = await db_get_llm_call_records(filter_model)
        return [LlmCallRecordType.from_pydantic(r) for r in records]

    @strawberry.field
    async def get_llm_call_records_by_run(self, run_id: str) -> list[LlmCallRecordType]:
        """Get all LLM call records for a specific run."""
        records = await db_get_llm_call_records_by_run(run_id)
        return [LlmCallRecordType.from_pydantic(r) for r in records]


# ============================================================================
# GraphQL Mutation
# ============================================================================


@strawberry.type
class StorageMutation:
    """GraphQL mutations for storage records."""

    @strawberry.mutation
    async def create_run_record(self, input_data: RunRecordInputType) -> RunRecordType:
        """Create a new run record."""
        pydantic_input = input_data.to_pydantic()
        record = await db_create_run_record(pydantic_input)
        return RunRecordType.from_pydantic(record)

    @strawberry.mutation
    async def update_run_record(
        self, run_id: str, input_data: RunRecordUpdateInput
    ) -> Optional[RunRecordType]:
        """Update an existing run record."""
        pydantic_input = input_data.to_pydantic()
        record = await db_update_run_record(run_id, pydantic_input)
        if record is None:
            return None
        return RunRecordType.from_pydantic(record)

    @strawberry.mutation
    async def create_llm_call_record(
        self, input_data: LlmCallRecordInputType
    ) -> LlmCallRecordType:
        """Create a new LLM call record."""
        pydantic_input = input_data.to_pydantic()
        record = await db_create_llm_call_record(pydantic_input)
        return LlmCallRecordType.from_pydantic(record)

    @strawberry.mutation
    async def update_llm_call_record(
        self, span_id: str, input_data: LlmCallRecordUpdateInput
    ) -> Optional[LlmCallRecordType]:
        """Update an existing LLM call record."""
        pydantic_input = input_data.to_pydantic()
        record = await db_update_llm_call_record(span_id, pydantic_input)
        if record is None:
            return None
        return LlmCallRecordType.from_pydantic(record)
