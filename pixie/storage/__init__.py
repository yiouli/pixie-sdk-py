"""Storage module for run and LLM call records."""

from .tables import (
    create_tables,
    RunRecord as RunRecordTable,
    LlmCallRecord as LlmCallRecordTable,
)
from .types import (
    Rating,
    RunSource,
    Message,
    RatingDetails,
    AppInfoRecord,
    SessionInfoRecord,
    PromptInfoRecord,
    RunRecord,
    RunRecordInput,
    RunRecordUpdate,
    LlmCallRecord,
    LlmCallRecordInput,
    LlmCallRecordUpdate,
    RecordFilters,
)
from .operations import (
    create_run_record,
    get_run_record,
    update_run_record,
    get_run_records,
    create_llm_call_record,
    get_llm_call_record,
    update_llm_call_record,
    get_llm_call_records,
    get_llm_call_records_by_run,
)

__all__ = [
    # Tables
    "create_tables",
    "RunRecordTable",
    "LlmCallRecordTable",
    # Types
    "Rating",
    "RunSource",
    "Message",
    "RatingDetails",
    "AppInfoRecord",
    "SessionInfoRecord",
    "PromptInfoRecord",
    "RunRecord",
    "RunRecordInput",
    "RunRecordUpdate",
    "LlmCallRecord",
    "LlmCallRecordInput",
    "LlmCallRecordUpdate",
    "RecordFilters",
    # Operations
    "create_run_record",
    "get_run_record",
    "update_run_record",
    "get_run_records",
    "create_llm_call_record",
    "get_llm_call_record",
    "update_llm_call_record",
    "get_llm_call_records",
    "get_llm_call_records_by_run",
]
